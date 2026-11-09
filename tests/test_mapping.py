"""Mapping tests: bounded, monotonic, deterministic."""

from __future__ import annotations

import pytest

from diastole.config import MappingConfig
from diastole.signal.mapping import BpmMapper, map_bpm


def test_mapping_is_monotonic() -> None:
    mapper = BpmMapper()
    values = []
    for bpm in range(30, 191, 5):
        p = map_bpm(float(bpm))
        values.append(p)
    for a, b in zip(values, values[1:]):
        assert b.tempo_bpm >= a.tempo_bpm
        assert b.energy >= a.energy
        assert b.rhythmic_density >= a.rhythmic_density
        assert b.note_activity >= a.note_activity
        assert b.dynamics >= a.dynamics
        assert b.rhythmic_intensity >= a.rhythmic_intensity


def test_mapping_bounds() -> None:
    cfg = MappingConfig()
    for bpm in (0, 30, 60, 100, 220, 400):
        p = map_bpm(bpm)
        assert cfg.tempo_min <= p.tempo_bpm <= cfg.tempo_max
        for value in (
            p.energy,
            p.rhythmic_density,
            p.note_activity,
            p.dynamics,
            p.rhythmic_intensity,
            p.arousal,
        ):
            assert 0.0 <= value <= 1.0
        assert p.beat_subdivision in (2, 4)


def test_expected_points() -> None:
    low = map_bpm(60)
    high = map_bpm(140)
    assert low.tempo_bpm == pytest.approx(69.0)
    assert high.tempo_bpm == pytest.approx(121.0)
    assert not low.drum_gate  # arousal ~0.029 -> intensity 0.172
    assert high.drum_gate
    assert low.beat_subdivision == 2
    assert high.beat_subdivision == 4


def test_mapping_is_deterministic() -> None:
    mapper = BpmMapper()
    p1 = map_bpm(97.5)
    p2 = map_bpm(97.5)
    assert p1 == p2
