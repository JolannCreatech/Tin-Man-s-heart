"""Canonical musical timeline, 16th-note scheduler and SMF writer.

The timeline is the single source of truth.  MIDI is written from it, and the
25 Hz conditioning adapter reads the *same* events.  Nothing is regenerated per
heart-rate sample; the scheduler only appends events for a short lookahead
window, so a session sounds like one continuous piece.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
import threading
from pathlib import Path

from diastole.config import DEFAULT_PPQ, NUM_MIDI_PITCHES, SchedulerConfig
from diastole.music.pitch import PitchProfile
from diastole.signal.mapping import MusicalParameters


@dataclasses.dataclass(frozen=True)
class NoteEvent:
    time: float
    pitch: int
    duration: float
    velocity: int = 90
    channel: int = 0

    @property
    def end_time(self) -> float:
        return self.time + self.duration


@dataclasses.dataclass(frozen=True)
class DrumEvent:
    time: float
    note: int
    velocity: int = 90
    channel: int = 9


@dataclasses.dataclass(frozen=True)
class TempoChange:
    time: float
    bpm: float


class MusicalTimeline:
    """Thread-safe collection of absolute-time musical events."""

    def __init__(self, ppq: int = DEFAULT_PPQ) -> None:
        self.ppq = int(ppq)
        self._notes: list[NoteEvent] = []
        self._drums: list[DrumEvent] = []
        self._tempos: list[TempoChange] = []
        self._lock = threading.RLock()

    # -- mutation ---------------------------------------------------------
    def set_tempo(self, time: float, bpm: float, force: bool = False) -> bool:
        if not math.isfinite(bpm) or bpm <= 0:
            return False
        with self._lock:
            if not self._tempos:
                self._tempos.append(TempoChange(float(time), float(bpm)))
                return True
            last = self._tempos[-1]
            if abs(last.bpm - bpm) < 0.25 and not force:
                return False
            if time < last.time - 1e-6:
                # A late tempo update replaces the last one in place rather
                # than corrupting the time map.
                self._tempos[-1] = TempoChange(float(max(0.0, time)), float(bpm))
                return True
            self._tempos.append(TempoChange(float(time), float(bpm)))
            return True

    def add_note(
        self, time: float, pitch: int, duration: float, velocity: int = 90, channel: int = 0
    ) -> NoteEvent:
        pitch = int(min(NUM_MIDI_PITCHES - 1, max(0, pitch)))
        event = NoteEvent(
            time=float(max(0.0, time)),
            pitch=pitch,
            duration=float(max(0.01, duration)),
            velocity=int(min(127, max(1, velocity))),
            channel=int(channel),
        )
        with self._lock:
            self._notes.append(event)
        return event

    def add_drum(self, time: float, note: int, velocity: int = 90) -> DrumEvent:
        event = DrumEvent(
            time=float(max(0.0, time)),
            note=int(min(127, max(0, note))),
            velocity=int(min(127, max(1, velocity))),
        )
        with self._lock:
            self._drums.append(event)
        return event

    # -- queries ----------------------------------------------------------
    @property
    def notes(self) -> tuple[NoteEvent, ...]:
        with self._lock:
            return tuple(self._notes)

    @property
    def drums(self) -> tuple[DrumEvent, ...]:
        with self._lock:
            return tuple(self._drums)

    @property
    def tempo_changes(self) -> tuple[TempoChange, ...]:
        with self._lock:
            return tuple(self._tempos)

    @property
    def note_count(self) -> int:
        with self._lock:
            return len(self._notes)

    @property
    def drum_count(self) -> int:
        with self._lock:
            return len(self._drums)

    def end_time(self) -> float:
        with self._lock:
            ends = [n.end_time for n in self._notes]
            ends += [d.time for d in self._drums]
        return max(ends, default=0.0)

    def notes_active_at(self, time: float) -> list[NoteEvent]:
        with self._lock:
            return [n for n in self._notes if n.time <= time < n.end_time]

    def note_onsets_between(self, start: float, end: float) -> list[NoteEvent]:
        with self._lock:
            return [n for n in self._notes if start <= n.time < end]

    def drums_between(self, start: float, end: float) -> list[DrumEvent]:
        with self._lock:
            return [d for d in self._drums if start <= d.time < end]

    def pitch_class_histogram(self) -> list[int]:
        counts = [0] * 12
        with self._lock:
            for note in self._notes:
                counts[note.pitch % 12] += 1
        return counts

    def tempo_at(self, time: float) -> float:
        with self._lock:
            bpm = 120.0
            for change in self._tempos:
                if change.time > time:
                    break
                bpm = change.bpm
        return bpm

    # -- MIDI -------------------------------------------------------------
    def _tempo_map(self) -> list[TempoChange]:
        with self._lock:
            changes = list(self._tempos)
        if not changes:
            changes = [TempoChange(0.0, 120.0)]
        changes.sort(key=lambda c: c.time)
        if changes[0].time > 0.0:
            changes.insert(0, TempoChange(0.0, changes[0].bpm))
        return changes

    def seconds_to_ticks(self, time: float) -> int:
        """Integrate the piecewise-constant tempo map up to ``time``."""

        changes = self._tempo_map()
        ticks = 0.0
        last_time = 0.0
        bpm = changes[0].bpm
        for change in changes:
            if change.time >= time:
                break
            ticks += (change.time - last_time) * bpm / 60.0 * self.ppq
            last_time = change.time
            bpm = change.bpm
        ticks += (max(0.0, time) - last_time) * bpm / 60.0 * self.ppq
        return int(round(ticks))

    def write_midi(self, path: str | Path) -> Path:
        """Atomically write the whole session as a type-1 SMF.

        The file is replaced on every call, so one session always has exactly
        one MIDI artifact that grows rather than a new file per measurement.
        """

        import mido

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ticks_per_beat = self.ppq

        midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name="DIASTOLE", time=0))

        scheduled: list[tuple[int, int, int, mido.Message]] = []
        seq = 0
        for change in self._tempo_map():
            scheduled.append(
                (
                    self.seconds_to_ticks(change.time),
                    1,
                    seq,
                    mido.MetaMessage(
                        "set_tempo", tempo=mido.bpm2tempo(change.bpm), time=0
                    ),
                )
            )
            seq += 1

        with self._lock:
            notes = list(self._notes)
            drums = list(self._drums)

        for note in notes:
            on_tick = self.seconds_to_ticks(note.time)
            off_tick = max(on_tick + 1, self.seconds_to_ticks(note.end_time))
            scheduled.append(
                (
                    off_tick,
                    0,
                    seq,
                    mido.Message(
                        "note_off",
                        note=note.pitch,
                        velocity=0,
                        channel=note.channel,
                        time=0,
                    ),
                )
            )
            seq += 1
            scheduled.append(
                (
                    on_tick,
                    2,
                    seq,
                    mido.Message(
                        "note_on",
                        note=note.pitch,
                        velocity=note.velocity,
                        channel=note.channel,
                        time=0,
                    ),
                )
            )
            seq += 1

        for drum in drums:
            on_tick = self.seconds_to_ticks(drum.time)
            off_tick = max(on_tick + 1, self.seconds_to_ticks(drum.time + 0.03))
            scheduled.append(
                (
                    off_tick,
                    0,
                    seq,
                    mido.Message(
                        "note_off", note=drum.note, velocity=0, channel=drum.channel, time=0
                    ),
                )
            )
            seq += 1
            scheduled.append(
                (
                    on_tick,
                    3,
                    seq,
                    mido.Message(
                        "note_on",
                        note=drum.note,
                        velocity=drum.velocity,
                        channel=drum.channel,
                        time=0,
                    ),
                )
            )
            seq += 1

        scheduled.sort(key=lambda item: (item[0], item[1], item[2]))
        last_tick = 0
        for tick, _priority, _seq, message in scheduled:
            message.time = max(0, tick - last_tick)
            track.append(message)
            last_tick = tick

        track.append(mido.MetaMessage("end_of_track", time=0))
        tmp = path.with_suffix(path.suffix + ".tmp")
        midi.save(str(tmp))
        os.replace(tmp, path)
        return path

    def summary(self) -> dict:
        changes = self.tempo_changes
        return {
            "note_events": self.note_count,
            "drum_events": self.drum_count,
            "tempo_changes": len(changes),
            "tempo_min": round(min((c.bpm for c in changes), default=0.0), 2),
            "tempo_max": round(max((c.bpm for c in changes), default=0.0), 2),
            "duration_seconds": round(self.end_time(), 3),
            "pitch_class_histogram": self.pitch_class_histogram(),
        }


class TimelineScheduler:
    """Append events on a 16th-note grid using the latest musical parameters."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        profile: PitchProfile | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.profile = profile or PitchProfile.fallback()
        base_seed = self.config.seed if seed is None else int(seed)
        self.rng_note = random.Random(base_seed)
        self.rng_drum = random.Random(base_seed + 1)
        self.rng_harmony = random.Random(base_seed + 2)
        self._next_step_time: float | None = None
        self._step_index = 0
        self._last_tempo: float | None = None
        self._chord: list[int] = []
        self._chord_root: int | None = None
        self._steps_at_current_chord = 0

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def next_step_time(self) -> float | None:
        return self._next_step_time

    @property
    def current_chord(self) -> list[int]:
        return list(self._chord)

    def step_duration(self, tempo_bpm: float) -> float:
        return 60.0 / max(1.0, tempo_bpm) / self.config.steps_per_beat

    def reset(self, now: float = 0.0) -> None:
        self._next_step_time = float(now)
        self._step_index = 0
        self._last_tempo = None
        self._chord = []
        self._chord_root = None
        self._steps_at_current_chord = 0

    def update(
        self,
        now: float,
        params: MusicalParameters,
        timeline: MusicalTimeline,
    ) -> int:
        """Schedule up to ``lookahead_seconds`` ahead; return events added."""

        now = float(now)
        tempo = float(params.tempo_bpm)
        if self._next_step_time is None or self._last_tempo is None:
            self._next_step_time = now
            self._last_tempo = tempo
            timeline.set_tempo(now, tempo, force=True)
        elif abs(tempo - self._last_tempo) >= 0.25:
            old_step = self.step_duration(self._last_tempo)
            new_step = self.step_duration(tempo)
            remaining = max(0.0, self._next_step_time - now)
            self._next_step_time = now + remaining * (new_step / old_step)
            self._last_tempo = tempo
            timeline.set_tempo(now, tempo)

        step = self.step_duration(tempo)

        # If inference stalled, do not dump a burst of past notes.
        if self._next_step_time < now - step:
            skipped = int((now - self._next_step_time) // step)
            self._next_step_time += skipped * step
            self._step_index += skipped

        added = 0
        horizon = now + self.config.lookahead_seconds
        while self._next_step_time <= horizon:
            added += self._schedule_step(self._next_step_time, params, timeline)
            self._next_step_time += step
            self._step_index += 1
        return added

    def _schedule_step(
        self, time: float, params: MusicalParameters, timeline: MusicalTimeline
    ) -> int:
        cfg = self.config
        pos = self._step_index % cfg.steps_per_bar
        beat_pos = pos % cfg.steps_per_beat
        added = 0

        if pos == 0:
            self._chord = self.profile.chord(
                self.rng_harmony, count=3, previous_root=self._chord_root
            )
            self._chord_root = self._chord[0] % 12 if self._chord else None
            self._steps_at_current_chord = 0
        self._steps_at_current_chord += 1

        # Position weight: downbeats speak, 16ths whisper.
        if pos == 0:
            weight = 1.0
        elif beat_pos == 0:
            weight = 0.85
        elif pos % 2 == 0:
            weight = 0.55
        else:
            weight = 0.30

        allow_note = True
        if params.beat_subdivision <= 2 and pos % 2 == 1:
            allow_note = False

        if allow_note and self.rng_note.random() < params.note_activity * weight:
            duration_steps = max(
                1.0,
                round((8.0 - 6.0 * params.energy) * self.rng_note.uniform(0.7, 1.25)),
            )
            duration = duration_steps * self.step_duration(params.tempo_bpm)
            velocity = int(
                min(127, max(30, round(36 + 88 * params.dynamics + self.rng_note.uniform(-6, 6))))
            )
            if pos == 0 and self._chord:
                pitches = self._chord
            elif self._chord:
                pitches = [self.profile.melody_pitch(self.rng_note)]
            else:
                pitches = [self.profile.melody_pitch(self.rng_note)]
            for pitch in pitches:
                timeline.add_note(time, pitch, duration, velocity=velocity)
                added += 1

        if params.drum_gate:
            added += self._schedule_drums(time, pos, params, timeline)

        return added

    def _schedule_drums(
        self, time: float, pos: int, params: MusicalParameters, timeline: MusicalTimeline
    ) -> int:
        rng = self.rng_drum
        intensity = params.rhythmic_intensity
        velocity = int(min(127, max(30, round(30 + 90 * params.dynamics))))
        added = 0

        # Kick on 1 and 3, snare on 2 and 4, hats on the 8th/16th grid.
        if pos in (0, 8) and rng.random() < intensity:
            timeline.add_drum(time, 36, velocity=velocity)
            added += 1
        if pos in (4, 12) and rng.random() < intensity * 0.9:
            timeline.add_drum(time, 38, velocity=velocity)
            added += 1
        if pos % 2 == 0 and rng.random() < intensity * 0.38:
            timeline.add_drum(time, 42, velocity=max(25, velocity - 22))
            added += 1
        if pos == 14 and rng.random() < intensity * 0.20:
            timeline.add_drum(time, 46, velocity=max(25, velocity - 10))
            added += 1
        return added
