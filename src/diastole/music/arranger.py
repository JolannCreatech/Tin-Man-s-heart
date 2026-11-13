"""BPM-locked deterministic arranger for the MIDI-derived material.

The arranger walks the *musical beat clock* rather than wall time: each 40 ms
generation frame advances the beat by ``tempo * FRAME_SECONDS / 60``.  Notes
that fall inside the frame are placed at their exact sub-frame sample offset,
so the accompaniment is sample-accurate even though the clock is advanced in
frame-sized steps.

The heartbeat is *not* scheduled here.  It is rendered by the playback thread
from the current BPM, so it keeps beating even if MRT2 inference stalls.  The
arranger only emits the MIDI/percussion documentation for the timeline.
"""

from __future__ import annotations

import dataclasses
import math

from diastole.config import FRAME_SECONDS, ArrangementConfig
from diastole.music.midi_analysis import MidiAnalysis
from diastole.music.midi_render import (
    LoopMaterial,
    chord_voicing,
    loop_material,
    pitch_near,
)


@dataclasses.dataclass(frozen=True)
class ArrangementEvent:
    """One deterministic accompaniment voice onset."""

    start_sample: int
    time: float
    kind: str  # bass | pad | lead
    pitch: int
    duration: float
    velocity: int

    @property
    def end_time(self) -> float:
        return self.time + self.duration


@dataclasses.dataclass(frozen=True)
class DrumOnset:
    time: float
    note: int
    velocity: int


@dataclasses.dataclass(frozen=True)
class _Onset:
    beat: float
    kind: str
    pitch: int
    duration_beats: float
    velocity: int


class MidiArranger:
    """Emit loop-relative musical events against a live BPM clock."""

    def __init__(
        self,
        analysis: MidiAnalysis,
        config: ArrangementConfig | None = None,
        *,
        sample_rate: int = 48_000,
    ) -> None:
        self.analysis = analysis
        self.config = config or ArrangementConfig()
        self.sample_rate = int(sample_rate)
        self.material: LoopMaterial = loop_material(analysis)
        self.loop_beats = max(1.0, self.material.loop_length_beats)
        self._beat = 0.0
        self._onsets = self._build_onsets()
        self._drums = self._build_drums()

    # -- material --------------------------------------------------------
    def _chord_at(self, rel_beat: float):
        start = self.material.loop_start_beat
        abs_beat = start + (rel_beat % self.loop_beats)
        for chord in self.material.chords:
            if chord.start_beat <= abs_beat < chord.end_beat:
                return chord
        return self.material.chords[0] if self.material.chords else None

    def _build_onsets(self) -> list[_Onset]:
        cfg = self.config
        bpb = max(1, int(cfg.beats_per_bar))
        onsets: list[_Onset] = []
        n_bars = max(1, int(round(self.loop_beats / bpb)))
        harmonic = max(0.5, float(cfg.harmonic_rhythm_beats))

        for bar in range(n_bars):
            bar_beat = bar * bpb
            if cfg.enable_pad:
                chord = self._chord_at(bar_beat)
                if chord is not None:
                    for pitch in chord_voicing(chord, center=60):
                        onsets.append(
                            _Onset(
                                bar_beat,
                                "pad",
                                pitch,
                                float(bpb),
                                int(40 + 40 * cfg.pad_density),
                            )
                        )
            if cfg.enable_bass:
                steps = max(1, int(round(bpb / harmonic)))
                for step in range(steps):
                    beat = bar_beat + step * harmonic
                    chord = self._chord_at(beat)
                    if chord is None or chord.bass_pitch is None:
                        continue
                    pitch = int(chord.bass_pitch)
                    if step % 2 == 1:
                        pitch = pitch_near((chord.root + 7) % 12, pitch + 7)
                    onsets.append(
                        _Onset(
                            beat,
                            "bass",
                            pitch,
                            harmonic * 0.95,
                            104 if step == 0 else 86,
                        )
                    )
        if cfg.enable_melody:
            for note in self.material.theme:
                rel = note.beat - self.material.loop_start_beat
                if 0.0 <= rel < self.loop_beats:
                    onsets.append(
                        _Onset(
                            float(rel),
                            "lead",
                            int(note.pitch),
                            float(note.duration_beats),
                            int(note.velocity),
                        )
                    )
        onsets.sort(key=lambda o: (o.beat, o.kind, o.pitch))
        return onsets

    def _build_drums(self) -> list[DrumOnset]:
        """Documentation of the heartbeat rhythm used by the playback synth."""

        cfg = self.config
        if not cfg.heartbeat:
            return []
        bpb = max(1, int(cfg.beats_per_bar))
        n_bars = max(1, int(round(self.loop_beats / bpb)))
        beats: list[DrumOnset] = []
        for bar in range(n_bars):
            for beat in range(bpb):
                abs_beat = bar * bpb + beat
                t = abs_beat * 60.0 / max(1.0, self.material.bpm)
                accent = beat % max(1, cfg.accent_every) == 0
                beats.append(DrumOnset(t, 36, 112 if accent else 92))
                if cfg.heartbeat_pattern == "heartbeat":
                    dub = abs_beat + 0.30
                    beats.append(
                        DrumOnset(
                            dub * 60.0 / max(1.0, self.material.bpm),
                            38,
                            74 if accent else 60,
                        )
                    )
                if cfg.heartbeat_subdivision >= 2:
                    for sub in range(1, cfg.heartbeat_subdivision):
                        ticks = abs_beat + sub / cfg.heartbeat_subdivision
                        beats.append(
                            DrumOnset(
                                ticks * 60.0 / max(1.0, self.material.bpm),
                                42,
                                46 if accent else 38,
                            )
                        )
        beats.sort(key=lambda d: d.time)
        return beats

    # -- clock -----------------------------------------------------------
    @property
    def beat(self) -> float:
        return self._beat

    def reset(self) -> None:
        self._beat = 0.0

    def update(
        self, audio_time: float, tempo_bpm: float
    ) -> tuple[list[ArrangementEvent], list[DrumOnset]]:
        """Return accompaniment events and drum documentation for one frame."""

        tempo = max(1.0, float(tempo_bpm))
        start_beat = self._beat
        delta = FRAME_SECONDS * tempo / 60.0
        end_beat = start_beat + delta
        sr = self.sample_rate
        events: list[ArrangementEvent] = []

        if delta > 0:
            for onset in self._onsets:
                # First occurrence at or after start_beat.
                k = math.floor((start_beat - onset.beat) / self.loop_beats)
                beat = onset.beat + k * self.loop_beats
                while beat < start_beat - 1e-9:
                    beat += self.loop_beats
                while beat < end_beat - 1e-9:
                    frac = min(1.0, max(0.0, (beat - start_beat) / delta))
                    time = audio_time + frac * FRAME_SECONDS
                    duration = onset.duration_beats * 60.0 / tempo
                    events.append(
                        ArrangementEvent(
                            start_sample=int(round(time * sr)),
                            time=time,
                            kind=onset.kind,
                            pitch=onset.pitch,
                            duration=duration,
                            velocity=onset.velocity,
                        )
                    )
                    beat += self.loop_beats

        drums: list[DrumOnset] = []
        if self._drums and delta > 0:
            loop_seconds = self.loop_beats * 60.0 / tempo
            for doc in self._drums:
                rel = doc.time
                k = math.floor((audio_time - rel) / loop_seconds)
                t = rel + k * loop_seconds
                while t < audio_time - 1e-9:
                    t += loop_seconds
                while t < audio_time + FRAME_SECONDS - 1e-9:
                    drums.append(DrumOnset(t, doc.note, doc.velocity))
                    t += loop_seconds

        self._beat = end_beat
        return events, drums
