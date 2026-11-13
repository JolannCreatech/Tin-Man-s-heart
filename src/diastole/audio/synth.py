"""Real-time deterministic synthesis.

Two layers live here:

* :class:`AccompanimentSynth` - pre-rendered bass/pad/lead voices pushed by the
  generation thread; the playback thread only adds scheduled slices.
* :class:`HeartbeatSynth` - a BPM-locked lub-dub heartbeat generated *inside*
  the playback thread.  It never depends on inference, so it stays stable even
  when MRT2 stalls.

Both are mono; the mixer handles stereo placement and gains.
"""

from __future__ import annotations

import bisect
import math
import threading

import numpy as np

from diastole.config import ArrangementConfig
from diastole.music.midi_render import render_voice


def _add_at(buffer: np.ndarray, signal: np.ndarray, start: int) -> None:
    if signal.size == 0:
        return
    start = int(start)
    end = min(buffer.shape[0], start + signal.shape[0])
    if end <= start:
        return
    buffer[start:end] += signal[: end - start]


class AccompanimentSynth:
    """Pending deterministic voices, rendered by the generation thread."""

    def __init__(self, sample_rate: int = 48_000, max_pending: int = 512) -> None:
        self.sample_rate = int(sample_rate)
        self.max_pending = int(max_pending)
        self._voices: list[tuple[int, np.ndarray]] = []
        self._starts: list[int] = []
        self._lock = threading.Lock()
        self.dropped = 0

    def add(
        self,
        start_sample: int,
        kind: str,
        pitch: int,
        duration: float,
        velocity: float,
    ) -> None:
        wave = render_voice(kind, pitch, duration, velocity, self.sample_rate)
        start = int(start_sample)
        with self._lock:
            index = bisect.bisect_left(self._starts, start)
            self._starts.insert(index, start)
            self._voices.insert(index, (start, wave))
            if len(self._voices) > self.max_pending:
                # Drop the oldest pending voice rather than growing without
                # bound if the playback thread is not consuming.
                self._starts.pop(0)
                self._voices.pop(0)
                self.dropped += 1

    def render(self, start_sample: int, frames: int) -> np.ndarray:
        frames = max(0, int(frames))
        out = np.zeros(frames, dtype=np.float32)
        if frames == 0:
            return out
        end = start_sample + frames
        with self._lock:
            keep: list[tuple[int, np.ndarray]] = []
            keep_starts: list[int] = []
            for start, wave in self._voices:
                voice_end = start + wave.shape[0]
                if voice_end <= start_sample:
                    continue
                if start < end:
                    lo = max(start_sample, start)
                    hi = min(end, voice_end)
                    if hi > lo:
                        out[lo - start_sample : hi - start_sample] += wave[
                            lo - start : hi - start
                        ]
                if voice_end > end:
                    keep.append((start, wave))
                    keep_starts.append(start)
            self._voices = keep
            self._starts = keep_starts
        return out

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._voices)

    def clear(self) -> None:
        with self._lock:
            self._voices.clear()
            self._starts.clear()


def _thump(
    sample_rate: int,
    *,
    f_start: float,
    f_end: float,
    decay: float,
    amplitude: float,
) -> np.ndarray:
    length = max(1, int(round(decay * 5.0 * sample_rate)))
    t = np.arange(length, dtype=np.float64) / sample_rate
    freq = f_end + (f_start - f_end) * np.exp(-t / 0.025)
    phase = 2.0 * math.pi * np.cumsum(freq) / sample_rate
    env = np.exp(-t / decay) * (1.0 - np.exp(-t / 0.004))
    return (amplitude * np.sin(phase) * env).astype(np.float32)


def _tick(sample_rate: int, amplitude: float) -> np.ndarray:
    decay = 0.012
    length = max(1, int(round(decay * 5.0 * sample_rate)))
    t = np.arange(length, dtype=np.float64) / sample_rate
    env = np.exp(-t / decay)
    return (amplitude * np.sin(2.0 * math.pi * 1400.0 * t) * env).astype(np.float32)


class HeartbeatSynth:
    """Sample-accurate, BPM-locked heartbeat/percussion layer."""

    def __init__(
        self,
        sample_rate: int = 48_000,
        config: ArrangementConfig | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.config = config or ArrangementConfig()
        self._bpm = 60.0
        self._beat = 0.0  # absolute musical beat position
        self._lock = threading.Lock()
        self._lub = _thump(
            self.sample_rate, f_start=92.0, f_end=44.0, decay=0.105, amplitude=0.92
        )
        self._dub = _thump(
            self.sample_rate, f_start=78.0, f_end=40.0, decay=0.085, amplitude=0.55
        )
        self._tick = _tick(self.sample_rate, 0.10)
        self._beat_events = 0

    def set_bpm(self, bpm: float) -> None:
        if math.isfinite(bpm) and bpm > 0:
            with self._lock:
                self._bpm = float(bpm)

    @property
    def bpm(self) -> float:
        with self._lock:
            return self._bpm

    @property
    def beat(self) -> float:
        with self._lock:
            return self._beat

    def reset(self) -> None:
        with self._lock:
            self._beat = 0.0

    def _pattern_offsets(self) -> list[tuple[float, str]]:
        pattern = self.config.heartbeat_pattern
        offsets: list[tuple[float, str]] = []
        if pattern == "heartbeat":
            offsets.append((0.0, "lub"))
            offsets.append((0.30, "dub"))
        elif pattern == "kick":
            offsets.append((0.0, "lub"))
        else:  # pulse
            offsets.append((0.0, "lub"))
        if self.config.heartbeat_subdivision >= 2:
            for sub in range(1, self.config.heartbeat_subdivision):
                offsets.append((sub / self.config.heartbeat_subdivision, "tick"))
        return offsets

    def render(self, start_sample: int, frames: int) -> np.ndarray:
        frames = max(0, int(frames))
        out = np.zeros(frames, dtype=np.float32)
        if frames == 0 or not self.config.heartbeat:
            return out
        with self._lock:
            bpm = self._bpm
            start = self._beat
        spb_samples = 60.0 / max(1.0, bpm) * self.sample_rate
        end = start + frames / max(1e-9, spb_samples)
        offsets = self._pattern_offsets()

        for offset, kind in offsets:
            k = math.floor(start - offset)
            beat = k + offset
            while beat < start - 1e-9:
                beat += 1.0
            while beat < end - 1e-9:
                sample = int(round((beat - start) * spb_samples))
                accent = abs(beat - round(beat)) < 1e-6 and (
                    int(round(beat)) % max(1, self.config.accent_every) == 0
                )
                if kind == "lub":
                    amp = 1.15 if accent else 0.95
                    _add_at(out, self._lub * amp, sample)
                    self._beat_events += 1
                elif kind == "dub":
                    amp = 0.75 if accent else 0.55
                    _add_at(out, self._dub * amp, sample)
                else:
                    _add_at(out, self._tick * (1.1 if accent else 0.7), sample)
                beat += 1.0

        with self._lock:
            self._beat = end
        return out

    @property
    def beat_events(self) -> int:
        return self._beat_events
