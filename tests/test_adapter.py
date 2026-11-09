"""Conditioning adapter tests."""

from __future__ import annotations

from diastole.music.adapter import FrameAdapter, validate_style_tokens
from diastole.music.timeline import MusicalTimeline


def test_onset_sustain_off_semantics() -> None:
    timeline = MusicalTimeline()
    timeline.add_note(0.0, 60, 0.1)
    adapter = FrameAdapter()

    f0 = adapter.frame(0.0, timeline)
    assert f0.notes[60] == 2
    assert 60 in f0.active_pitches
    assert 60 in f0.onset_pitches

    f1 = adapter.frame(0.04, timeline)
    assert f1.notes[60] == 1
    assert 60 not in f1.onset_pitches

    f2 = adapter.frame(0.12, timeline)
    assert f2.notes[60] == 0


def test_unused_notes_default_to_zero_or_mask() -> None:
    timeline = MusicalTimeline()
    adapter = FrameAdapter(mask_unused_notes=False)
    assert adapter.frame(0.0, timeline).notes[42] == 0
    masked = FrameAdapter(mask_unused_notes=True)
    assert masked.frame(0.0, timeline).notes[42] == -1


def test_drum_hit_window() -> None:
    timeline = MusicalTimeline()
    timeline.add_drum(0.05, 36)
    adapter = FrameAdapter()
    assert adapter.frame(0.04, timeline).drum == 1
    assert adapter.frame(0.04, timeline).drum_hit is True
    assert adapter.frame(0.08, timeline).drum == 0


def test_frame_shape_and_style_validation() -> None:
    timeline = MusicalTimeline()
    timeline.add_note(0.0, 64, 0.2)
    adapter = FrameAdapter()
    cond = adapter.conditioning(0.0, timeline, [1, 2, 3])
    assert len(cond.style_tokens) == 12
    assert len(cond.frame.notes) == 128
    assert isinstance(cond.frame.drum, int)
    assert validate_style_tokens(None) == [-1] * 12
    assert len(validate_style_tokens(list(range(20)))) == 12
