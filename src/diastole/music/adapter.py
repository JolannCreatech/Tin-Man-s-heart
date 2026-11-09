"""25 Hz adapter: musical events -> MRT2 conditioning tokens.

MRT2 does not read MIDI.  This module samples the same ``MusicalTimeline`` that
is written to the session MIDI file and produces the 128-value onset pianoroll
and one drum token per 40 ms frame:

  -1 masked, 0 off, 1 sustaining, 2 onset, 3 free (notes)
  -1 masked, 0 no hit, 1 hit (drums)
"""

from __future__ import annotations

import dataclasses

from diastole.config import (
    DRUM_TOKEN_COUNT,
    FRAME_SECONDS,
    NOTE_TOKEN_COUNT,
    STYLE_TOKEN_COUNT,
)
from diastole.music.timeline import MusicalTimeline


@dataclasses.dataclass(frozen=True)
class ConditioningFrame:
    notes: tuple[int, ...]
    drum: int
    active_pitches: tuple[int, ...] = ()
    onset_pitches: tuple[int, ...] = ()
    drum_hit: bool = False

    def as_dict(self) -> dict:
        return {
            "notes": list(self.notes),
            "drum": self.drum,
            "active_pitches": list(self.active_pitches),
            "onset_pitches": list(self.onset_pitches),
            "drum_hit": self.drum_hit,
        }


@dataclasses.dataclass(frozen=True)
class ConditioningSet:
    """Everything the engine needs for one generation step."""

    style_tokens: tuple[int, ...]
    frame: ConditioningFrame

    def as_dict(self) -> dict:
        return {
            "style_tokens": list(self.style_tokens),
            "frame": self.frame.as_dict(),
        }


def validate_style_tokens(tokens: list[int] | tuple[int, ...] | None) -> list[int]:
    if tokens is None:
        return [-1] * STYLE_TOKEN_COUNT
    values = [int(t) for t in tokens]
    if len(values) < STYLE_TOKEN_COUNT:
        values += [-1] * (STYLE_TOKEN_COUNT - len(values))
    return values[:STYLE_TOKEN_COUNT]


class FrameAdapter:
    """Convert timeline events into one conditioning frame."""

    def __init__(self, mask_unused_notes: bool = False) -> None:
        self.mask_unused_notes = bool(mask_unused_notes)

    def frame(
        self,
        time: float,
        timeline: MusicalTimeline,
        frame_duration: float = FRAME_SECONDS,
    ) -> ConditioningFrame:
        end = time + frame_duration
        active = timeline.notes_active_at(time)
        onsets = timeline.note_onsets_between(time, end)

        default = -1 if self.mask_unused_notes else 0
        notes = [default] * NOTE_TOKEN_COUNT
        for note in active:
            notes[note.pitch] = 1
        for note in onsets:
            notes[note.pitch] = 2

        drums = timeline.drums_between(time, end)
        drum = 1 if drums else 0
        if not drums and self.mask_unused_notes:
            # Still explicit: 0 means "do not play a drum now"; -1 would hand
            # the decision back to the model.
            drum = 0

        return ConditioningFrame(
            notes=tuple(notes),
            drum=int(drum) if DRUM_TOKEN_COUNT == 1 else int(drum),
            active_pitches=tuple(sorted(n.pitch for n in active)),
            onset_pitches=tuple(sorted(n.pitch for n in onsets)),
            drum_hit=bool(drums),
        )

    def conditioning(
        self,
        time: float,
        timeline: MusicalTimeline,
        style_tokens: list[int] | tuple[int, ...] | None,
        frame_duration: float = FRAME_SECONDS,
    ) -> ConditioningSet:
        return ConditioningSet(
            style_tokens=tuple(validate_style_tokens(style_tokens)),
            frame=self.frame(time, timeline, frame_duration),
        )
