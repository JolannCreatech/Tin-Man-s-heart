"""Instrumental rendering of the MIDI-derived musical material.

The transcription is not a score, so this module does not try to reproduce it
literally.  It renders a *clean instrumental arrangement* from the analysed
loop (bass, chord pad, lead theme) with a small self-contained numpy synth.
The result is used for two things:

1. conditioning: MusicCoCa style tokens and the SpectroStream prefill are
   derived from an instrumental rendering instead of the vocal-heavy MP3;
2. validation: the rendered loop can be auditioned on its own, and it is the
   musical reference the live arrangement follows.

No FluidSynth or SoundFont is required.  Everything is deterministic.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from diastole import config as cfg
from diastole.config import ArrangementConfig
from diastole.music.midi_analysis import (
    ChordEvent,
    MidiAnalysis,
    ThemeNote,
    pc_name,
)


@dataclasses.dataclass(frozen=True)
class LoopMaterial:
    """Loop-relative musical material used by the renderer and the arranger."""

    loop_start_beat: float
    loop_length_beats: float
    bpm: float
    chords: tuple[ChordEvent, ...]
    theme: tuple[ThemeNote, ...]

    @property
    def loop_seconds(self) -> float:
        return self.loop_length_beats * 60.0 / max(1.0, self.bpm)


def loop_material(analysis: MidiAnalysis) -> LoopMaterial:
    start = analysis.loop_start_beat
    end = analysis.loop_end_beat
    chords = tuple(
        c for c in analysis.chords if c.end_beat > start and c.start_beat < end
    )
    theme = tuple(
        t
        for t in analysis.theme
        if start <= t.beat < end
    )
    return LoopMaterial(
        loop_start_beat=start,
        loop_length_beats=analysis.loop_length_beats,
        bpm=analysis.bpm,
        chords=chords,
        theme=theme,
    )


# ---------------------------------------------------------------------------
# Small synthesis primitives
# ---------------------------------------------------------------------------


def _adsr(
    n: int,
    sr: int,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
) -> np.ndarray:
    n = max(1, int(n))
    a = min(n, int(max(1, attack * sr)))
    d = min(n - a, int(max(0, decay * sr)))
    r = min(n - a - d, int(max(1, release * sr)))
    s = max(0, n - a - d - r)
    env = np.empty(n, dtype=np.float32)
    pos = 0
    if a:
        env[pos : pos + a] = np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32)
        pos += a
    if d:
        env[pos : pos + d] = np.linspace(
            1.0, sustain, d, endpoint=False, dtype=np.float32
        )
        pos += d
    if s:
        env[pos : pos + s] = sustain
        pos += s
    if r:
        env[pos : pos + r] = np.linspace(
            sustain, 0.0, r, endpoint=False, dtype=np.float32
        )
        pos += r
    if pos < n:
        env[pos:] = 0.0
    return env


def _phase(freq: float, n: int, sr: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    return (2.0 * math.pi * freq * t).astype(np.float32)


def _freq(pitch: int) -> float:
    return 440.0 * 2.0 ** ((int(pitch) - 69) / 12.0)


def _bass_voice(pitch: int, n: int, sr: int, velocity: float) -> np.ndarray:
    env = _adsr(n, sr, attack=0.008, decay=0.18, sustain=0.62, release=0.14)
    ph = _phase(_freq(pitch), n, sr)
    tone = (
        np.sin(ph)
        + 0.28 * np.sin(2.0 * ph)
        + 0.10 * np.sin(3.0 * ph)
        + 0.05 * np.sin(4.0 * ph)
    ).astype(np.float32)
    return (0.5 * velocity * tone * env).astype(np.float32)


def _pad_voice(pitch: int, n: int, sr: int, velocity: float) -> np.ndarray:
    env = _adsr(n, sr, attack=0.35, decay=0.45, sustain=0.70, release=0.55)
    ph = _phase(_freq(pitch), n, sr)
    # Two slightly detuned partials make a soft, wide string-like tone.
    detune = 2.0 * math.pi * 0.8 * np.arange(n, dtype=np.float64) / sr
    tone = (
        np.sin(ph + detune.astype(np.float32))
        + np.sin(ph - detune.astype(np.float32))
        + 0.22 * np.sin(2.0 * ph)
        + 0.08 * np.sin(3.0 * ph)
    ).astype(np.float32)
    return (0.16 * velocity * tone * env).astype(np.float32)


def _lead_voice(pitch: int, n: int, sr: int, velocity: float) -> np.ndarray:
    env = _adsr(n, sr, attack=0.012, decay=0.20, sustain=0.45, release=0.22)
    # Slight exponential decay on top of the sustain makes a plucked feel.
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    env = env * (0.65 + 0.35 * np.exp(-3.0 * x)).astype(np.float32)
    ph = _phase(_freq(pitch), n, sr)
    tone = (
        np.sin(ph)
        + 0.24 * np.sin(2.0 * ph)
        + 0.10 * np.sin(3.0 * ph)
        + 0.04 * np.sin(4.0 * ph)
    ).astype(np.float32)
    return (0.30 * velocity * tone * env).astype(np.float32)


def _add(buffer: np.ndarray, signal: np.ndarray, start: int) -> None:
    if signal.size == 0:
        return
    start = max(0, int(start))
    end = min(buffer.shape[0], start + signal.shape[0])
    if end <= start:
        return
    buffer[start:end] += signal[: end - start]


def _chord_voicing(chord: ChordEvent, center: int = 60) -> list[int]:
    """Spread the chord tones around ``center`` without muddiness."""

    classes = chord.pitch_classes
    root = chord.root
    voices: list[int] = []
    for interval, pc in zip((0, 3, 6), classes):
        target = _pitch_near(pc, center + interval)
        voices.append(target)
    # Double the root one octave up for a fuller pad, but keep it in range.
    if max(voices) < 69:
        voices.append(min(84, voices[0] + 12))
    return sorted(set(voices))[:4]


def _pitch_near(pc: int, center: int) -> int:
    lo = max(0, int(center) - 6)
    hi = min(127, int(center) + 6)
    candidates = [p for p in range(lo, hi + 1) if p % 12 == int(pc) % 12]
    if not candidates:
        return int(center)
    return int(min(candidates, key=lambda p: abs(p - center)))


# Public helpers reused by the live arranger.
def chord_voicing(chord: ChordEvent, center: int = 60) -> list[int]:
    return _chord_voicing(chord, center)


def pitch_near(pc: int, center: int) -> int:
    return _pitch_near(pc, center)


def _reverb(x: np.ndarray, sr: int, decay: float, mix: float, seed: int = 0) -> np.ndarray:
    """Cheap stereo reverb using a synthetic decaying-noise impulse response."""

    if mix <= 0:
        return x
    n = max(1, int(decay * sr))
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sr
    shape = np.exp(-t * (5.0 / max(0.1, decay)))
    irs = []
    for _ in range(2):
        ir = rng.standard_normal(n) * shape
        energy = float(np.sqrt(np.sum(ir * ir)))
        irs.append((ir / max(1e-9, energy)).astype(np.float32))
    ir = np.stack(irs, axis=1)
    wet = fftconvolve(x, ir, mode="full", axes=0)[: x.shape[0]]
    return ((1.0 - mix) * x + mix * wet).astype(np.float32)


# ---------------------------------------------------------------------------
# Arrangement rendering
# ---------------------------------------------------------------------------


def render_voice(
    kind: str,
    pitch: int,
    duration: float,
    velocity: float,
    sample_rate: int,
) -> np.ndarray:
    """Render one deterministic voice as a mono float32 buffer."""

    n = max(1, int(round(max(0.02, float(duration)) * int(sample_rate))))
    velocity = float(min(1.0, max(0.0, velocity)))
    if kind == "bass":
        return _bass_voice(int(pitch), n, int(sample_rate), velocity)
    if kind == "pad":
        return _pad_voice(int(pitch), n, int(sample_rate), velocity)
    return _lead_voice(int(pitch), n, int(sample_rate), velocity)


def render_arrangement(
    analysis: MidiAnalysis,
    *,
    seconds: float,
    sample_rate: int = cfg.SAMPLE_RATE,
    config: ArrangementConfig | None = None,
    reverb_mix: float = 0.16,
    seed: int = 0,
) -> np.ndarray:
    """Render the analysed loop as stereo float32 instrumental audio."""

    config = config or ArrangementConfig()
    material = loop_material(analysis)
    spb = 60.0 / max(1.0, material.bpm)
    loop_beats = max(1.0, material.loop_length_beats)
    loop_seconds = loop_beats * spb
    n = max(1, int(round(float(seconds) * sample_rate)))
    iterations = max(1, int(math.ceil(float(seconds) / max(1e-6, loop_seconds))))

    bass = np.zeros(n, dtype=np.float32)
    pad = np.zeros(n, dtype=np.float32)
    lead = np.zeros(n, dtype=np.float32)

    for iteration in range(iterations):
        base_beat = iteration * loop_beats
        for chord in material.chords:
            rel_start = max(chord.start_beat, material.loop_start_beat)
            rel_end = min(chord.end_beat, material.loop_start_beat + loop_beats)
            if rel_end <= rel_start:
                continue
            start_beat = base_beat + (rel_start - material.loop_start_beat)
            duration_beats = rel_end - rel_start
            start = int(round(start_beat * spb * sample_rate))
            if start >= n:
                continue

            # Bass: retrigger on each harmonic-rhythm unit for movement.
            pulse = max(1, int(config.harmonic_rhythm_beats))
            if config.enable_bass and chord.bass_pitch is not None:
                steps = max(1, int(math.ceil(duration_beats / pulse)))
                for step in range(steps):
                    pitch = int(chord.bass_pitch)
                    if step > 0 and step % 2 == 1:
                        pitch = _pitch_near(
                            (chord.root + 7) % 12, int(chord.bass_pitch) + 7
                        )
                    seg_start = start + int(round(step * pulse * spb * sample_rate))
                    seg_beats = min(pulse, duration_beats - step * pulse)
                    if seg_beats <= 0:
                        continue
                    length = int(round(seg_beats * spb * sample_rate * 0.98))
                    velocity = 0.95 if step == 0 else 0.72
                    _add(bass, _bass_voice(pitch, length, sample_rate, velocity), seg_start)

            if config.enable_pad:
                length = int(round(max(0.1, duration_beats * spb) * sample_rate))
                for pitch in _chord_voicing(chord):
                    _add(
                        pad,
                        _pad_voice(pitch, length, sample_rate, config.pad_density),
                        start,
                    )

        if config.enable_melody:
            for note in material.theme:
                rel = note.beat - material.loop_start_beat
                if rel < 0 or rel >= loop_beats:
                    continue
                start_beat = base_beat + rel
                start = int(round(start_beat * spb * sample_rate))
                if start >= n:
                    continue
                length = int(
                    round(max(0.08, note.duration_beats * spb * 0.98) * sample_rate)
                )
                velocity = (note.velocity / 127.0) * config.melody_density
                _add(lead, _lead_voice(note.pitch, length, sample_rate, velocity), start)

    mix = (
        bass * 0.95
        + pad * 0.85
        + lead * 0.75
    ).astype(np.float32)
    stereo = np.stack([mix, mix], axis=1)
    stereo = _reverb(stereo, sample_rate, decay=1.8, mix=reverb_mix, seed=seed)

    # Gentle fade at both ends to avoid clicks when cached and reused.
    fade = min(n // 2, int(0.08 * sample_rate))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        stereo[:fade] *= ramp
        stereo[-fade:] *= ramp[::-1]

    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 0:
        stereo = stereo * (0.89 / peak)
    return np.ascontiguousarray(stereo, dtype=np.float32)


def render_cache_path(
    analysis: MidiAnalysis,
    *,
    seconds: float,
    sample_rate: int = cfg.SAMPLE_RATE,
    config: ArrangementConfig | None = None,
    directory: str | Path | None = None,
) -> Path:
    config = config or ArrangementConfig()
    directory = Path(directory or (cfg.DEFAULT_CACHE_DIR / "reference"))
    material = loop_material(analysis)
    digest = hashlib.sha1()
    for part in (
        analysis.source,
        round(float(analysis.duration_seconds), 3),
        int(analysis.loop_start_bar),
        int(analysis.loop_bars),
        round(float(material.loop_length_beats), 3),
        round(float(seconds), 3),
        int(sample_rate),
        config.mode,
        int(config.enable_bass),
        int(config.enable_pad),
        int(config.enable_melody),
        config.harmonic_rhythm_beats,
    ):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return directory / f"midi_render_{digest.hexdigest()[:16]}.wav"


def load_or_render(
    analysis: MidiAnalysis,
    *,
    seconds: float = 48.0,
    sample_rate: int = cfg.SAMPLE_RATE,
    config: ArrangementConfig | None = None,
    directory: str | Path | None = None,
    force: bool = False,
    progress: callable | None = None,
) -> tuple[np.ndarray, Path]:
    """Return (audio, path); render and cache on first use."""

    import soundfile as sf

    path = render_cache_path(
        analysis,
        seconds=seconds,
        sample_rate=sample_rate,
        config=config,
        directory=directory,
    )
    if path.exists() and not force:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.concatenate([data, data], axis=1)
        if int(sr) == int(sample_rate):
            return np.ascontiguousarray(data, dtype=np.float32), path
    if progress:
        progress("Rendering instrumental MIDI loop (first run only)...")
    audio = render_arrangement(
        analysis,
        seconds=seconds,
        sample_rate=sample_rate,
        config=config,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, int(sample_rate), subtype="PCM_16")
    if progress:
        progress(f"Rendered {audio.shape[0] / sample_rate:.1f}s -> {path.name}")
    return audio, path


def summary(analysis: MidiAnalysis) -> dict:
    material = loop_material(analysis)
    return {
        "loop_start_beat": round(material.loop_start_beat, 3),
        "loop_length_beats": round(material.loop_length_beats, 3),
        "loop_seconds": round(material.loop_seconds, 3),
        "loop_chords": [
            {
                "start": round(max(c.start_beat, material.loop_start_beat), 3),
                "end": round(min(c.end_beat, material.loop_start_beat + material.loop_length_beats), 3),
                "symbol": c.symbol,
                "root": pc_name(c.root),
                "confidence": round(c.confidence, 3),
            }
            for c in material.chords
            if c.end_beat > material.loop_start_beat
            and c.start_beat < material.loop_start_beat + material.loop_length_beats
        ],
        "theme_notes": len(material.theme),
    }
