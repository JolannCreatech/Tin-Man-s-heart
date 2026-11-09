"""Heart-rate signal processing.

Pipeline: validate -> reject stale/outliers -> median -> slew-limit -> EMA.
The output is a small immutable state consumed by the musical mapping.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from collections import deque

from diastole.config import ProcessorConfig
from diastole.sensors.base import PhysiologicalSample


def smoothstep(lo: float, hi: float, x: float) -> float:
    """C1-continuous 0..1 ramp, clamped outside [lo, hi]."""

    if hi <= lo:
        return 0.0 if x < lo else 1.0
    t = max(0.0, min(1.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


@dataclasses.dataclass(frozen=True)
class PhysiologicalState:
    """Smoothed physiological state used by the mapping layer."""

    timestamp: float
    raw_bpm: float
    bpm: float
    trend_bpm_per_s: float
    variability: float
    arousal: float
    valid: bool
    sample_age: float
    source: str = "none"

    @classmethod
    def fallback(cls, timestamp: float | None = None, bpm: float = 60.0) -> "PhysiologicalState":
        return cls(
            timestamp=timestamp if timestamp is not None else time.monotonic(),
            raw_bpm=bpm,
            bpm=bpm,
            trend_bpm_per_s=0.0,
            variability=0.0,
            arousal=smoothstep(50.0, 150.0, bpm),
            valid=False,
            sample_age=math.inf,
            source="fallback",
        )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class BpmProcessor:
    """Deterministic, bounded smoothing of noisy heart-rate samples."""

    def __init__(self, config: ProcessorConfig | None = None) -> None:
        self.config = config or ProcessorConfig()
        self._samples: deque[tuple[float, float]] = deque(maxlen=max(3, self.config.median_window))
        self._history: deque[float] = deque(maxlen=max(3, self.config.variability_window))
        self._state: PhysiologicalState | None = None
        self._last_accepted_time: float | None = None
        self._last_sample_timestamp: float | None = None
        self._smoothed: float | None = None
        self._trend = 0.0

    @property
    def state(self) -> PhysiologicalState | None:
        return self._state

    def reset(self, bpm: float | None = None, now: float | None = None) -> None:
        self._samples.clear()
        self._history.clear()
        self._state = None
        self._last_accepted_time = None
        self._last_sample_timestamp = None
        self._smoothed = bpm
        self._trend = 0.0
        if bpm is not None:
            self._state = PhysiologicalState.fallback(
                now if now is not None else time.monotonic(), bpm
            )

    def update(
        self, sample: PhysiologicalSample | None, now: float | None = None
    ) -> PhysiologicalState:
        """Ingest one sample and return the current smoothed state.

        Invalid or stale samples never move the estimate; they only mark the
        returned state as not valid so the UI can show a disconnection.
        """

        now = time.monotonic() if now is None else float(now)
        cfg = self.config

        if sample is None:
            return self._invalid(now, math.inf, "none", "no sample")

        age = max(0.0, now - sample.timestamp)
        if not sample.valid:
            return self._invalid(now, age, sample.source, "invalid sample")
        if not math.isfinite(sample.bpm):
            return self._invalid(now, age, sample.source, "non-finite BPM")
        if not (cfg.min_bpm <= sample.bpm <= cfg.max_bpm):
            return self._invalid(now, age, sample.source, "out of range")
        if age > cfg.max_sample_age:
            return self._invalid(now, age, sample.source, "stale sample")

        # The generation loop polls faster than the sensor updates.  Ingesting
        # the same timestamp repeatedly would let one frozen reading dominate
        # the median and the slew-limited EMA, so only genuinely new samples
        # move the estimate.
        if (
            self._last_sample_timestamp is not None
            and sample.timestamp <= self._last_sample_timestamp
        ):
            if self._state is not None:
                self._state = dataclasses.replace(
                    self._state, timestamp=now, sample_age=age, valid=True
                )
                return self._state
            return self._invalid(now, age, sample.source, "duplicate sample")
        self._last_sample_timestamp = float(sample.timestamp)

        # Median of the most recent accepted readings suppresses spikes.
        self._samples.append((now, float(sample.bpm)))
        cutoff = now - max(cfg.max_sample_age * 2.0, 1.0)
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        median_bpm = statistics.median(v for _, v in self._samples)

        if self._smoothed is None or self._last_accepted_time is None:
            self._smoothed = median_bpm
            self._last_accepted_time = now
            self._trend = 0.0
        else:
            dt = max(1e-4, min(0.5, now - self._last_accepted_time))
            alpha = 1.0 - math.exp(-dt / max(1e-4, cfg.ema_tau))
            previous = self._smoothed
            # EMA follows the median, then the *output* is slew-limited.  This
            # keeps rapid scenario changes expressive while never stepping the
            # musical grid faster than the configured BPM/s limit.
            ema_target = previous + alpha * (median_bpm - previous)
            max_delta = cfg.max_slew_bpm_per_s * dt
            self._smoothed = min(
                previous + max_delta, max(previous - max_delta, ema_target)
            )
            raw_trend = (self._smoothed - previous) / dt
            trend_alpha = 1.0 - math.exp(-dt / 1.6)
            self._trend = self._trend + trend_alpha * (raw_trend - self._trend)
            self._last_accepted_time = now

        self._history.append(float(sample.bpm))
        variability = (
            statistics.pstdev(self._history) if len(self._history) >= 2 else 0.0
        )
        bpm = float(self._smoothed)
        self._state = PhysiologicalState(
            timestamp=now,
            raw_bpm=float(sample.bpm),
            bpm=bpm,
            trend_bpm_per_s=float(self._trend),
            variability=float(variability),
            arousal=smoothstep(cfg.arousal_low, cfg.arousal_high, bpm),
            valid=True,
            sample_age=age,
            source=sample.source,
        )
        return self._state

    def _invalid(
        self, now: float, age: float, source: str, _reason: str
    ) -> PhysiologicalState:
        if self._state is None:
            self._state = PhysiologicalState.fallback(now)
            return self._state
        # Keep the last good value but report that it is no longer trustworthy.
        s = self._state
        self._state = dataclasses.replace(
            s,
            timestamp=now,
            valid=False,
            sample_age=age,
            source=source if source != "none" else s.source,
        )
        return self._state
