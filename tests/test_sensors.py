"""Simulated sensor tests."""

from __future__ import annotations

import time

import pytest

from diastole.sensors.simulated import SimulatedHeartRateSensor, scenario_profile


def test_scenario_shapes() -> None:
    assert scenario_profile("rest", 10) == pytest.approx(62)
    assert scenario_profile("sudden", 2) == pytest.approx(62)
    assert scenario_profile("sudden", 12) == pytest.approx(142)
    assert scenario_profile("demo", 0) == pytest.approx(62)
    assert scenario_profile("demo", 20) == pytest.approx(132)
    assert scenario_profile("demo", 40) == pytest.approx(64)
    assert scenario_profile("gradual", 10) > scenario_profile("gradual", 1)


def test_manual_override() -> None:
    sensor = SimulatedHeartRateSensor(scenario="rest", seed=1)
    sensor.set_manual_bpm(111)
    assert sensor.bpm_at(5.0) == pytest.approx(111)
    sensor.set_manual_bpm(None)
    assert sensor.bpm_at(0.0) != pytest.approx(111)


def test_invalid_manual_bpm() -> None:
    sensor = SimulatedHeartRateSensor()
    with pytest.raises(ValueError):
        sensor.set_manual_bpm(10)
    with pytest.raises(ValueError):
        sensor.set_scenario("not-a-scenario")


def test_thread_emits_monotonic_samples() -> None:
    sensor = SimulatedHeartRateSensor(scenario="rest", update_hz=20, seed=3)
    sensor.start()
    try:
        first = sensor.read()
        assert first is not None
        time.sleep(0.3)
        second = sensor.read()
        assert second is not None
        assert second.timestamp >= first.timestamp
        assert 30 <= second.bpm <= 220
        assert sensor.sample_count >= 2
    finally:
        sensor.stop()


def test_scenario_change_resets_origins() -> None:
    sensor = SimulatedHeartRateSensor(scenario="rest", seed=2)
    sensor.set_manual_bpm(100)
    sensor.set_scenario("sudden")
    assert sensor.bpm_at(0.0) != pytest.approx(100)
