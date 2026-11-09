"""Engine conditioning layout and mock-engine tests (no model load)."""

from __future__ import annotations

import numpy as np
import pytest

from diastole.config import EngineConfig, NOTE_TOKEN_COUNT, STYLE_TOKEN_COUNT
from diastole.generation.engine import MRT2Engine, MockEngine, _as_conditioning


def test_conditioning_width_and_validation() -> None:
    raw = _as_conditioning([-1] * 12, [0] * 128, [0])
    assert raw.shape == (141,)
    with pytest.raises(ValueError):
        _as_conditioning([0] * 11, [0] * 128, [0])
    with pytest.raises(ValueError):
        _as_conditioning([0] * 12, [0] * 127, [0])


def test_build_args_offsets_and_negative_masks() -> None:
    engine = object.__new__(MRT2Engine)
    engine.config = EngineConfig()
    style = [1] * STYLE_TOKEN_COUNT
    notes = [0] * NOTE_TOKEN_COUNT
    notes[60] = 2
    drums = [1]
    args = engine._build_args(style, notes, drums)
    positive = np.asarray(args[0])
    neg_style = np.asarray(args[6])
    neg_notes = np.asarray(args[7])
    assert positive.shape == (1, 1, 141)
    assert positive[0, 0, 0] == 1 + 7
    assert positive[0, 0, 12 + 60] == 2 + 7
    assert positive[0, 0, 140] == 1 + 7
    # Negating style masks only the 12 style slots.
    assert np.all(neg_style[0, 0, :12] == -1 + 7)
    assert np.all(neg_style[0, 0, 12:141] == positive[0, 0, 12:141])
    # Negating notes masks only the 128 note slots.
    assert np.all(neg_notes[0, 0, :12] == positive[0, 0, :12])
    assert np.all(neg_notes[0, 0, 12:140] == -1 + 7)
    assert np.all(neg_notes[0, 0, 140:] == positive[0, 0, 140:])


def test_mock_engine_continuity_and_shapes() -> None:
    engine = MockEngine()
    style = [-1] * 12
    notes = [0] * 128
    notes[60] = 2
    a, ms = engine.step(style, notes, (0,))
    notes2 = [0] * 128
    notes2[60] = 1
    b, _ = engine.step(style, notes2, (0,))
    assert a.shape == (engine.samples_per_frame, 2)
    assert b.shape == a.shape
    assert np.isfinite(a).all()
    assert np.max(np.abs(a)) > 0
    assert engine.step_count == 2
    assert ms >= 0
