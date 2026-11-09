"""Deterministic physiological -> musical mapping.

MRT2 has no tempo knob: its clock is fixed at 25 Hz.  Tempo is therefore
expressed through *event scheduling* (beat period and which 16th-note grid
positions receive onsets), and arousal controls density, velocity, drum
activity and subdivision.  Every parameter below is bounded and monotonic in
BPM.
"""

from __future__ import annotations

import dataclasses
import math

from diastole.config import MappingConfig
from diastole.signal.processing import PhysiologicalState


@dataclasses.dataclass(frozen=True)
class MusicalParameters:
    """Bounded, serialisable musical target for one update."""

    tempo_bpm: float
    energy: float
    rhythmic_density: float
    note_activity: float
    dynamics: float
    rhythmic_intensity: float
    drum_gate: bool
    beat_subdivision: int
    arousal: float

    def clamp01(self) -> "MusicalParameters":
        def c(x: float) -> float:
            return min(1.0, max(0.0, float(x)))

        return dataclasses.replace(
            self,
            energy=c(self.energy),
            rhythmic_density=c(self.rhythmic_density),
            note_activity=c(self.note_activity),
            dynamics=c(self.dynamics),
            rhythmic_intensity=c(self.rhythmic_intensity),
            beat_subdivision=max(1, int(self.beat_subdivision)),
        )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


class BpmMapper:
    """Map a smoothed BPM state to musical parameters.

    The default mapping (documented in PLAN.md) is:
      tempo            = 30 + 0.65*bpm          (clamped 50..180)
      energy           = arousal
      density          = 0.20 + 0.65*arousal
      note activity    = 0.25 + 0.60*arousal
      dynamics         = 0.45 + 0.50*arousal
      drum intensity   = 0.15 + 0.75*arousal
      drum gate        = intensity >= 0.28
      subdivision      = 2 below arousal 0.5, else 4
    """

    def __init__(self, config: MappingConfig | None = None) -> None:
        self.config = config or MappingConfig()

    def map(self, state: PhysiologicalState) -> MusicalParameters:
        cfg = self.config
        bpm = float(state.bpm)
        if not math.isfinite(bpm):
            bpm = 60.0
        arousal = min(1.0, max(0.0, state.arousal))

        tempo = cfg.tempo_base + cfg.tempo_gain * bpm
        tempo = min(cfg.tempo_max, max(cfg.tempo_min, tempo))

        density = cfg.density_base + cfg.density_gain * arousal
        note_activity = cfg.note_activity_base + cfg.note_activity_gain * arousal
        dynamics = cfg.dynamics_base + cfg.dynamics_gain * arousal
        intensity = cfg.intensity_base + cfg.intensity_gain * arousal
        subdivision = (
            cfg.subdivision_high if arousal >= 0.5 else cfg.subdivision_low
        )

        return MusicalParameters(
            tempo_bpm=float(tempo),
            energy=float(arousal),
            rhythmic_density=float(density),
            note_activity=float(note_activity),
            dynamics=float(dynamics),
            rhythmic_intensity=float(intensity),
            drum_gate=bool(intensity >= cfg.drum_gate_threshold),
            beat_subdivision=int(subdivision),
            arousal=float(arousal),
        ).clamp01()


def map_bpm(bpm: float, mapping: MappingConfig | None = None) -> MusicalParameters:
    """Convenience helper used by tests and the report."""

    cfg = mapping or MappingConfig()
    state = PhysiologicalState(
        timestamp=0.0,
        raw_bpm=bpm,
        bpm=bpm,
        trend_bpm_per_s=0.0,
        variability=0.0,
        arousal=0.0,
        valid=True,
        sample_age=0.0,
    )
    # Reuse the canonical smoothstep/arousal bounds so this helper cannot drift
    # from runtime.
    from diastole.config import ProcessorConfig
    from diastole.signal.processing import smoothstep

    pc = ProcessorConfig()
    state = dataclasses.replace(
        state, arousal=smoothstep(pc.arousal_low, pc.arousal_high, bpm)
    )
    return BpmMapper(cfg).map(state)
