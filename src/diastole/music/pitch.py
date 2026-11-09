"""Pitch material derived from the reference transcription.

The generated control score should be harmonically related to
*If I Only Had a Heart*, so pitch classes are sampled from a duration-weighted
histogram of the existing MIDI transcription.  A fixed fallback (C major) is
used when the MIDI file is missing or empty.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np

FALLBACK_WEIGHTS = np.array(
    [1.0, 0.0, 0.8, 0.0, 0.9, 0.8, 0.0, 1.0, 0.0, 0.7, 0.0, 0.5],
    dtype=np.float64,
)


def _normalise(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(12)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return FALLBACK_WEIGHTS / FALLBACK_WEIGHTS.sum()
    return weights / total


@dataclasses.dataclass(frozen=True)
class PitchProfile:
    """Duration-weighted pitch-class distribution plus a playing register."""

    weights: np.ndarray
    register_low: int = 48
    register_high: int = 76
    source: str = "fallback"

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _normalise(self.weights))
        object.__setattr__(self, "register_low", int(max(0, min(127, self.register_low))))
        object.__setattr__(self, "register_high", int(max(self.register_low, min(127, self.register_high))))

    @classmethod
    def fallback(cls, register_low: int = 48, register_high: int = 76) -> "PitchProfile":
        return cls(
            weights=FALLBACK_WEIGHTS.copy(),
            register_low=register_low,
            register_high=register_high,
            source="fallback",
        )

    @classmethod
    def from_midi(
        cls,
        path: str | Path,
        register_low: int = 48,
        register_high: int = 76,
    ) -> "PitchProfile":
        path = Path(path)
        if not path.exists():
            return cls.fallback(register_low, register_high)

        import mido

        try:
            midi = mido.MidiFile(str(path))
        except Exception:
            return cls.fallback(register_low, register_high)

        weights = np.zeros(12, dtype=np.float64)
        active: dict[tuple[int, int], int] = {}
        tick = 0
        note_count = 0
        for msg in mido.merge_tracks(midi.tracks):
            tick += int(msg.time)
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = tick
            elif msg.type in ("note_off", "note_on"):
                key = (msg.channel, msg.note)
                start = active.pop(key, None)
                if start is None:
                    continue
                duration = max(1, tick - start)
                weights[msg.note % 12] += duration
                note_count += 1

        if note_count == 0 or weights.sum() <= 0:
            return cls.fallback(register_low, register_high)
        return cls(
            weights=_normalise(weights),
            register_low=register_low,
            register_high=register_high,
            source=str(path),
        )

    # -- generation -------------------------------------------------------
    def choose_pitch_class(self, rng: random.Random | np.random.Generator) -> int:
        p = self.weights
        if isinstance(rng, np.random.Generator):
            return int(rng.choice(12, p=p))
        return int(rng.choices(range(12), weights=p.tolist(), k=1)[0])

    def pc_to_midi(self, pitch_class: int, center: int | None = None) -> int:
        """Place a pitch class as close as possible to the register centre."""

        lo, hi = self.register_low, self.register_high
        center = (lo + hi) // 2 if center is None else int(center)
        candidates = [p for p in range(lo, hi + 1) if p % 12 == pitch_class % 12]
        if not candidates:
            return int(min(127, max(0, center)))
        return int(min(candidates, key=lambda p: abs(p - center)))

    def chord(
        self,
        rng: random.Random | np.random.Generator,
        count: int = 3,
        previous_root: int | None = None,
    ) -> list[int]:
        """Sample a harmonically plausible chord from the profile.

        The root avoids repeating the previous root when possible.  The third
        is major or minor depending on which pitch class the reference actually
        uses more, so the harmony stays tied to the song rather than to a
        hard-coded key.
        """

        root = self.choose_pitch_class(rng)
        if previous_root is not None and root == previous_root:
            order = np.argsort(-self.weights)
            alternatives = [int(p) for p in order if int(p) != root]
            if alternatives:
                root = alternatives[0]

        major_third = (root + 4) % 12
        minor_third = (root + 3) % 12
        third = major_third if self.weights[major_third] >= self.weights[minor_third] else minor_third
        intervals = [0]
        if count >= 2:
            intervals.append(7)
        if count >= 3:
            intervals.append(third - root)
        if count >= 4:
            intervals.append(12)
        pcs = [root + i for i in intervals]
        notes = sorted({self.pc_to_midi(pc % 12) for pc in pcs})
        return notes[: max(1, count)]

    def melody_pitch(self, rng: random.Random | np.random.Generator) -> int:
        return self.pc_to_midi(self.choose_pitch_class(rng))

    def as_dict(self) -> dict:
        return {
            "weights": {str(i): round(float(w), 5) for i, w in enumerate(self.weights)},
            "top_pitch_classes": [int(i) for i in np.argsort(-self.weights)[:5]],
            "register_low": self.register_low,
            "register_high": self.register_high,
            "source": self.source,
        }
