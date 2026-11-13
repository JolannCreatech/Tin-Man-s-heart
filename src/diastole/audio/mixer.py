"""Multi-layer mixer and real-time layer controller.

The playback thread calls :meth:`RealtimeLayers.process` once per block.  It
renders the deterministic heartbeat and accompaniment at the current sample
position, then mixes:

    master * (heartbeat*gain + accompaniment*gain + MRT2*gain
              + optional reference*gain)
    -> limiter -> clip

Gains are smoothed, and the limiter uses instantaneous attack with a smooth
release so the output cannot clip.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from diastole.config import ArrangementConfig, MixConfig, SAMPLE_RATE
from diastole.audio.synth import AccompanimentSynth, HeartbeatSynth


class SmoothedGain:
    """One-pole gain smoother that can be read block by block."""

    def __init__(self, target: float, sample_rate: int, seconds: float) -> None:
        self.sample_rate = int(sample_rate)
        self.target = float(target)
        self.current = float(target)
        self.decay = (
            math.exp(-1.0 / max(1.0, float(seconds) * self.sample_rate))
            if seconds > 0
            else 0.0
        )

    def set(self, target: float) -> None:
        self.target = max(0.0, float(target))

    def process(self, frames: int) -> np.ndarray:
        frames = max(0, int(frames))
        if frames == 0:
            return np.zeros(0, dtype=np.float32)
        if self.decay <= 0.0:
            self.current = self.target
            return np.full(frames, self.target, dtype=np.float32)
        ramp = self.decay ** np.arange(frames, dtype=np.float64)
        values = self.target + (self.current - self.target) * ramp
        self.current = float(values[-1])
        return values.astype(np.float32)


class Limiter:
    """Instant-attack / smooth-release limiter with a hard safety clip."""

    def __init__(
        self, threshold: float, sample_rate: int, release_seconds: float
    ) -> None:
        self.threshold = float(min(0.999, max(0.05, threshold)))
        self.sample_rate = int(sample_rate)
        self.release = max(1e-4, float(release_seconds))
        self.gain = 1.0
        self.reductions = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        frames = x.shape[0]
        env = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
        desired = np.minimum(1.0, self.threshold / np.maximum(env, 1e-9))
        alpha = 1.0 - math.exp(-1.0 / (self.release * self.sample_rate))
        gain = np.empty(frames, dtype=np.float64)
        g = self.gain
        reduced = False
        for i in range(frames):
            d = float(desired[i])
            if d < g:
                g = d
                reduced = True
            else:
                g += (d - g) * alpha
            gain[i] = g
        self.gain = g
        if reduced:
            self.reductions += 1
        if x.ndim == 1:
            return (x * gain.astype(np.float32)).astype(np.float32)
        return (x * gain[:, None].astype(np.float32)).astype(np.float32)


class LayerMixer:
    """Per-layer gains, optional ducking, limiter and reference playback."""

    def __init__(self, config: MixConfig | None = None, sample_rate: int = SAMPLE_RATE) -> None:
        self.config = config or MixConfig()
        self.sample_rate = int(sample_rate)
        smooth = self.config.smooth_gain_seconds
        self.gain_heart = SmoothedGain(self.config.heartbeat_gain, sample_rate, smooth)
        self.gain_accomp = SmoothedGain(
            self.config.accompaniment_gain, sample_rate, smooth
        )
        self.gain_mrt2 = SmoothedGain(self.config.mrt2_gain, sample_rate, smooth)
        self.gain_reference = SmoothedGain(
            self.config.reference_gain, sample_rate, smooth
        )
        self.gain_master = SmoothedGain(self.config.master_gain, sample_rate, smooth)
        self.limiter = Limiter(
            self.config.limiter_threshold,
            sample_rate,
            self.config.limiter_release_seconds,
        )
        self._duck_env = 0.0
        self._duck_decay = math.exp(
            -1.0 / max(1e-4, self.config.duck_release_seconds * sample_rate)
        )
        self._reference: np.ndarray | None = None
        self.peak = 0.0
        self.clipped = 0

    def set_reference(self, audio: np.ndarray | None) -> None:
        if audio is None:
            self._reference = None
            return
        data = np.asarray(audio, dtype=np.float32)
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        if data.shape[1] == 1:
            data = np.concatenate([data, data], axis=1)
        self._reference = np.ascontiguousarray(data[:, :2])

    def _reference_block(self, start_sample: int, frames: int) -> np.ndarray | None:
        if self._reference is None or self._reference.shape[0] == 0:
            return None
        length = self._reference.shape[0]
        idx = (start_sample + np.arange(frames)) % length
        return self._reference[idx]

    def process(
        self,
        mrt2: np.ndarray,
        heartbeat: np.ndarray,
        accompaniment: np.ndarray,
        start_sample: int = 0,
    ) -> np.ndarray:
        mrt2 = np.asarray(mrt2, dtype=np.float32)
        if mrt2.ndim == 1:
            mrt2 = np.stack([mrt2, mrt2], axis=1)
        frames = mrt2.shape[0]
        if frames == 0:
            return mrt2

        heart = np.asarray(heartbeat, dtype=np.float32).reshape(-1, 1)
        accomp = np.asarray(accompaniment, dtype=np.float32).reshape(-1, 1)
        if heart.shape[0] < frames:
            heart = np.pad(heart, ((0, frames - heart.shape[0]), (0, 0)))
        if accomp.shape[0] < frames:
            accomp = np.pad(accomp, ((0, frames - accomp.shape[0]), (0, 0)))

        g_heart = self.gain_heart.process(frames)[:, None]
        g_accomp = self.gain_accomp.process(frames)[:, None]
        g_mrt2 = self.gain_mrt2.process(frames)[:, None]
        g_master = self.gain_master.process(frames)[:, None]

        # Optional sidechain ducking of MRT2/accompaniment by the heartbeat.
        if self.config.ducking > 0:
            env = np.abs(heart[:, 0])
            duck = np.empty(frames, dtype=np.float32)
            e = self._duck_env
            for i in range(frames):
                e = max(float(env[i]), e * self._duck_decay)
                duck[i] = 1.0 - self.config.ducking * min(1.0, e / 0.5)
            self._duck_env = e
            duck = duck[:, None]
        else:
            duck = 1.0

        mixed = (
            mrt2 * g_mrt2 * duck
            + heart * g_heart
            + accomp * g_accomp * duck
        )
        ref = self._reference_block(start_sample, frames)
        if ref is not None:
            mixed = mixed + ref * self.gain_reference.process(frames)[:, None]

        mixed = mixed * g_master
        mixed = self.limiter.process(mixed)
        np.clip(mixed, -1.0, 1.0, out=mixed)
        if mixed.size:
            self.peak = max(self.peak, float(np.max(np.abs(mixed))))
        return np.ascontiguousarray(mixed, dtype=np.float32)


class RealtimeLayers:
    """Everything the playback thread needs to render the non-MRT2 layers."""

    def __init__(
        self,
        mix_config: MixConfig | None = None,
        arrangement_config: ArrangementConfig | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.mix_config = mix_config or MixConfig()
        self.arrangement_config = arrangement_config or ArrangementConfig()
        self.sample_rate = int(sample_rate)
        self.heartbeat = HeartbeatSynth(sample_rate, self.arrangement_config)
        self.accompaniment = AccompanimentSynth(sample_rate)
        self.mixer = LayerMixer(self.mix_config, sample_rate)

    def set_bpm(self, bpm: float) -> None:
        self.heartbeat.set_bpm(bpm)

    def add_accompaniment(
        self, start_sample: int, kind: str, pitch: int, duration: float, velocity: float
    ) -> None:
        self.accompaniment.add(start_sample, kind, pitch, duration, velocity)

    def process(self, block: np.ndarray, start_sample: int) -> np.ndarray:
        frames = int(np.asarray(block).shape[0])
        heart = self.heartbeat.render(start_sample, frames)
        accomp = self.accompaniment.render(start_sample, frames)
        return self.mixer.process(block, heart, accomp, start_sample=start_sample)

    def stats(self) -> dict:
        return {
            "heartbeat_gain": round(self.mix_config.heartbeat_gain, 3),
            "accompaniment_gain": round(self.mix_config.accompaniment_gain, 3),
            "mrt2_gain": round(self.mix_config.mrt2_gain, 3),
            "reference_gain": round(self.mix_config.reference_gain, 3),
            "master_gain": round(self.mix_config.master_gain, 3),
            "ducking": round(self.mix_config.ducking, 3),
            "pending_voices": self.accompaniment.pending,
            "dropped_voices": self.accompaniment.dropped,
            "limiter_reductions": self.mixer.limiter.reductions,
            "mix_peak": round(self.mixer.peak, 4),
            "clipped_samples": self.mixer.clipped,
            "heartbeat_bpm": round(self.heartbeat.bpm, 2),
            "heartbeat_beat": round(self.heartbeat.beat, 3),
            "heartbeat_beat_events": self.heartbeat.beat_events,
        }
