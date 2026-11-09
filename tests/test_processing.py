"""Signal-processing tests: median, slew, staleness, arousal."""

from __future__ import annotations

import math

import pytest

from diastole.config import ProcessorConfig
from diastole.sensors.base import PhysiologicalSample
from diastole.signal.processing import BpmProcessor, smoothstep


def sample(bpm: float, t: float, valid: bool = True) -> PhysiologicalSample:
    return PhysiologicalSample(bpm=bpm, timestamp=t, valid=valid)


def test_smoothstep_bounds_and_midpoint() -> None:
    assert smoothstep(50, 150, 0) == 0.0
    assert smoothstep(50, 150, 100) == pytest.approx(0.5)
    assert smoothstep(50, 150, 1000) == 1.0
    assert 0 <= smoothstep(50, 150, 77) <= 1


def test_first_sample_initialises_state() -> None:
    proc = BpmProcessor()
    state = proc.update(sample(70, 0.0), now=0.0)
    assert state.valid
    assert state.bpm == pytest.approx(70)
    assert state.raw_bpm == pytest.approx(70)
    assert state.arousal == pytest.approx(smoothstep(50, 150, 70))


def test_outlier_is_rejected_by_range() -> None:
    proc = BpmProcessor()
    state = proc.update(sample(64, 0.0), now=0.0)
    state = proc.update(sample(400, 0.25), now=0.25)
    assert state.valid is False
    assert state.bpm == pytest.approx(64)
    state = proc.update(sample(65, 0.5), now=0.5)
    assert state.valid
    assert 60 <= state.bpm <= 70


def test_spike_is_suppressed_by_median() -> None:
    cfg = ProcessorConfig(median_window=5, ema_tau=0.05, max_slew_bpm_per_s=1000)
    proc = BpmProcessor(cfg)
    t = 0.0
    for bpm in (62, 62, 62):
        proc.update(sample(bpm, t), now=t)
        t += 0.25
    state = proc.update(sample(140, t), now=t)  # one spike among five
    assert state.bpm < 75
    t += 0.25
    state = proc.update(sample(62, t), now=t)
    assert state.bpm < 75


def test_stale_sample_is_invalid_but_keeps_last_value() -> None:
    cfg = ProcessorConfig(max_sample_age=1.0, ema_tau=0.1)
    proc = BpmProcessor(cfg)
    proc.update(sample(70, 0.0), now=0.0)
    state = proc.update(sample(70, 0.0), now=5.0)
    assert state.valid is False
    assert state.bpm == pytest.approx(70)
    assert state.sample_age >= 4.9


def test_slew_limits_step_response() -> None:
    cfg = ProcessorConfig(
        ema_tau=0.2, max_slew_bpm_per_s=10.0, median_window=3
    )
    proc = BpmProcessor(cfg)
    t = 0.0
    for _ in range(4):
        state = proc.update(sample(60, t), now=t)
        t += 0.25
    before = state.bpm
    # A single 0.25 s update may move at most 2.5 BPM regardless of the target.
    state = proc.update(sample(140, t), now=t)
    assert state.bpm - before <= 2.5 + 1e-6


def test_trend_and_variability_are_finite() -> None:
    proc = BpmProcessor(ProcessorConfig(ema_tau=0.1, max_slew_bpm_per_s=1000))
    t = 0.0
    for bpm in (60, 61, 63, 66, 70):
        state = proc.update(sample(bpm, t), now=t)
        t += 0.25
    assert math.isfinite(state.trend_bpm_per_s)
    assert math.isfinite(state.variability)
    assert state.variability > 0


def test_duplicate_timestamp_does_not_move_estimate() -> None:
    cfg = ProcessorConfig(ema_tau=0.1, max_slew_bpm_per_s=1000, median_window=3)
    proc = BpmProcessor(cfg)
    proc.update(sample(60, 0.0), now=0.0)
    proc.update(sample(120, 0.25), now=0.25)
    moved = proc.state.bpm
    # Repeated polling of the same measurement must not pull the estimate.
    for now in (0.26, 0.27, 0.30):
        state = proc.update(sample(120, 0.25), now=now)
        assert state.valid
        assert state.bpm == pytest.approx(moved)


def test_reset_and_fallback() -> None:
    proc = BpmProcessor()
    state = proc.update(None, now=1.0)
    assert not state.valid
    assert state.bpm == pytest.approx(60.0)
    proc.reset(bpm=75, now=2.0)
    assert proc.state is not None
    assert proc.state.bpm == pytest.approx(75)
