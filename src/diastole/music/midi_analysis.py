"""MIDI analysis and preprocessing for the musical reference.

The file in ``music/`` is a *Basic Pitch* transcription of the full recording,
not an authoritative score.  It has a single track/program (electric piano),
contains the vocal line and the accompaniment mixed together, and its upper
register is noisy.  This module therefore treats the transcription as evidence
rather than truth:

  * key and tempo are estimated from duration-weighted pitch content;
  * the bass line (reliable) anchors a key-constrained chord progression;
  * a cleaned, continuity-smoothed upper voice yields the melodic theme;
  * a coherent 16-bar loop is selected automatically and reused for the
    arrangement and for rendering an instrumental conditioning signal.

Everything is deterministic and serialisable so the report can quote exactly
which musical material was used.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

PC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

# Krumhansl-Kessler key profiles.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Diatonic triad templates in a major key, by scale degree.
_MAJOR_DEGREES = (
    (0, "maj", (0, 4, 7)),
    (2, "min", (0, 3, 7)),
    (4, "min", (0, 3, 7)),
    (5, "maj", (0, 4, 7)),
    (7, "maj", (0, 4, 7)),
    (9, "min", (0, 3, 7)),
    (11, "dim", (0, 3, 6)),
)

# Functional transition preferences between scale degrees (0=I, 1=ii, ...).
_DEGREE_TRANSITIONS = np.array(
    [
        [0.32, 0.02, -0.15, 0.14, 0.16, 0.10, -0.20],
        [0.12, 0.10, 0.02, 0.08, 0.22, 0.04, -0.10],
        [0.10, 0.05, 0.10, 0.06, 0.04, 0.14, -0.10],
        [0.16, 0.06, -0.04, 0.16, 0.10, 0.10, -0.05],
        [0.30, 0.04, -0.08, 0.10, 0.14, 0.12, -0.10],
        [0.10, 0.12, 0.04, 0.16, 0.12, 0.10, -0.05],
        [0.16, 0.06, 0.02, 0.06, 0.12, 0.06, 0.00],
    ]
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def pc_name(pitch: int) -> str:
    return PC_NAMES[int(pitch) % 12]


def pitch_name(pitch: int) -> str:
    return f"{pc_name(pitch)}{int(pitch) // 12 - 1}"


@dataclasses.dataclass(frozen=True)
class MidiNote:
    start: float
    end: float
    pitch: int
    velocity: int
    channel: int
    track: int
    program: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_dict(self) -> dict:
        return {
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "pitch": int(self.pitch),
            "name": pitch_name(self.pitch),
            "velocity": int(self.velocity),
            "channel": int(self.channel),
            "track": int(self.track),
            "program": int(self.program),
        }


@dataclasses.dataclass(frozen=True)
class ChordEvent:
    start_beat: float
    end_beat: float
    root: int
    quality: str
    degree: int
    confidence: float
    bass_pitch: int | None = None

    @property
    def symbol(self) -> str:
        suffix = {"maj": "", "min": "m", "dim": "dim"}.get(self.quality, self.quality)
        return f"{pc_name(self.root)}{suffix}"

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        intervals = {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6)}[
            self.quality
        ]
        return tuple((self.root + i) % 12 for i in intervals)

    def as_dict(self) -> dict:
        return {
            "start_beat": round(self.start_beat, 3),
            "end_beat": round(self.end_beat, 3),
            "root": int(self.root),
            "quality": self.quality,
            "degree": int(self.degree),
            "symbol": self.symbol,
            "pitch_classes": list(self.pitch_classes),
            "bass_pitch": self.bass_pitch,
            "bass_name": pitch_name(self.bass_pitch) if self.bass_pitch is not None else None,
            "confidence": round(self.confidence, 4),
        }


@dataclasses.dataclass(frozen=True)
class BarInfo:
    index: int
    start_beat: float
    end_beat: float
    chord: str
    bass_pitch: int | None
    melody_pitch: int | None
    note_count: int
    density_hz: float

    def as_dict(self) -> dict:
        return {
            "index": int(self.index),
            "start_beat": round(self.start_beat, 3),
            "end_beat": round(self.end_beat, 3),
            "chord": self.chord,
            "bass": pitch_name(self.bass_pitch) if self.bass_pitch is not None else None,
            "melody": pitch_name(self.melody_pitch) if self.melody_pitch is not None else None,
            "note_count": int(self.note_count),
            "density_hz": round(self.density_hz, 3),
        }


@dataclasses.dataclass(frozen=True)
class ThemeNote:
    beat: float
    duration_beats: float
    pitch: int
    velocity: int

    def as_dict(self) -> dict:
        return {
            "beat": round(self.beat, 3),
            "duration_beats": round(self.duration_beats, 3),
            "pitch": int(self.pitch),
            "name": pitch_name(self.pitch),
            "velocity": int(self.velocity),
        }


@dataclasses.dataclass(frozen=True)
class MidiAnalysis:
    source: str
    duration_seconds: float
    duration_beats: float
    bpm: float
    ticks_per_beat: int
    time_signature: tuple[int, int]
    beats_per_bar: float
    key_tonic: int
    key_mode: str
    key_confidence: float
    key_correlations: dict[str, float]
    tracks: list[dict]
    programs: list[int]
    raw_note_count: int
    notes: list[MidiNote]
    chords: list[ChordEvent]
    bars: list[BarInfo]
    theme: list[ThemeNote]
    loop_start_bar: int
    loop_bars: int
    loop_start_beat: float
    loop_end_beat: float
    issues: dict
    stats: dict

    @property
    def key_name(self) -> str:
        return f"{PC_NAMES[self.key_tonic]} {self.key_mode}"

    @property
    def loop_length_beats(self) -> float:
        return max(1.0, self.loop_end_beat - self.loop_start_beat)

    def chord_at_beat(self, beat: float) -> ChordEvent | None:
        """Return the chord segment containing an absolute beat."""

        for chord in self.chords:
            if chord.start_beat <= beat < chord.end_beat:
                return chord
        if self.chords:
            return self.chords[-1]
        return None

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 3),
            "duration_beats": round(self.duration_beats, 3),
            "bpm": round(self.bpm, 3),
            "ticks_per_beat": int(self.ticks_per_beat),
            "time_signature": list(self.time_signature),
            "beats_per_bar": self.beats_per_bar,
            "key": {
                "tonic": int(self.key_tonic),
                "mode": self.key_mode,
                "name": self.key_name,
                "confidence": round(self.key_confidence, 4),
                "correlations": {k: round(v, 4) for k, v in self.key_correlations.items()},
            },
            "tracks": self.tracks,
            "programs": self.programs,
            "raw_note_count": self.raw_note_count,
            "note_count": len(self.notes),
            "notes": [n.as_dict() for n in self.notes],
            "chords": [c.as_dict() for c in self.chords],
            "bars": [b.as_dict() for b in self.bars],
            "theme": [t.as_dict() for t in self.theme],
            "loop": {
                "start_bar": int(self.loop_start_bar),
                "bars": int(self.loop_bars),
                "start_beat": round(self.loop_start_beat, 3),
                "end_beat": round(self.loop_end_beat, 3),
                "length_beats": round(self.loop_length_beats, 3),
            },
            "issues": self.issues,
            "stats": self.stats,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path


def _tempo_map(midi: object) -> list[tuple[int, int]]:
    changes: list[tuple[int, int]] = []
    for track in midi.tracks:  # type: ignore[attr-defined]
        tick = 0
        for msg in track:
            tick += int(msg.time)
            if msg.type == "set_tempo":
                changes.append((tick, int(msg.tempo)))
    changes.sort(key=lambda item: item[0])
    if not changes or changes[0][0] > 0:
        changes.insert(0, (0, 500_000))
    # Collapse duplicates on the same tick.
    unique: list[tuple[int, int]] = []
    for tick, tempo in changes:
        if unique and unique[-1][0] == tick:
            unique[-1] = (tick, tempo)
        else:
            unique.append((tick, tempo))
    return unique


class _TempoConverter:
    def __init__(
        self, changes: list[tuple[int, int]], ppq: int, end_tick: int | None = None
    ) -> None:
        self.ppq = max(1, int(ppq))
        self.changes = changes
        self.end_tick = max(1, int(end_tick) if end_tick else self.ppq * 4)
        self.ticks = [c[0] for c in changes]
        self.seconds_at = [0.0]
        for i in range(1, len(changes)):
            prev_tick, prev_tempo = changes[i - 1]
            cur_tick = changes[i][0]
            self.seconds_at.append(
                self.seconds_at[-1]
                + (cur_tick - prev_tick) / self.ppq * (prev_tempo / 1_000_000.0)
            )

    def seconds(self, tick: float) -> float:
        idx = bisect.bisect_right(self.ticks, tick) - 1
        idx = max(0, min(idx, len(self.changes) - 1))
        base_tick, tempo = self.changes[idx]
        return (
            self.seconds_at[idx]
            + (tick - base_tick) / self.ppq * (tempo / 1_000_000.0)
        )

    def beats(self, seconds: float) -> float:
        # Invert numerically; the map is tiny and monotonic.
        lo, hi = 0.0, float(max(self.end_tick, self.ticks[-1] + self.ppq))
        for _ in range(48):
            mid = (lo + hi) * 0.5
            if self.seconds(mid) < seconds:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5 / self.ppq


def _parse_notes(
    midi: object, converter: _TempoConverter
) -> tuple[list[MidiNote], list[dict], list[int], tuple[int, int], int]:
    notes: list[MidiNote] = []
    tracks_info: list[dict] = []
    programs: set[int] = set()
    signature = (4, 4)
    raw = 0

    for track_index, track in enumerate(midi.tracks):  # type: ignore[attr-defined]
        tick = 0
        program = 0
        active: dict[tuple[int, int], deque[tuple[int, int]]] = {}
        channel_programs: dict[int, int] = {}
        track_notes = 0
        for msg in track:
            tick += int(msg.time)
            if msg.type == "set_tempo":
                continue
            if msg.type == "time_signature":
                signature = (int(msg.numerator), int(msg.denominator))
                continue
            if msg.type == "program_change":
                program = int(msg.program)
                channel_programs[int(msg.channel)] = program
                programs.add(program)
                continue
            if msg.type == "note_on" and int(msg.velocity) > 0:
                raw += 1
                track_notes += 1
                key = (int(msg.channel), int(msg.note))
                active.setdefault(key, deque()).append((tick, int(msg.velocity)))
                continue
            if msg.type == "note_off" or (
                msg.type == "note_on" and int(msg.velocity) == 0
            ):
                key = (int(msg.channel), int(msg.note))
                queue = active.get(key)
                if not queue:
                    continue
                start_tick, velocity = queue.popleft()
                end_tick = max(start_tick + 1, tick)
                if int(msg.channel) == 9:
                    continue
                notes.append(
                    MidiNote(
                        start=converter.seconds(start_tick),
                        end=converter.seconds(end_tick),
                        pitch=int(msg.note),
                        velocity=int(velocity),
                        channel=int(msg.channel),
                        track=int(track_index),
                        program=int(channel_programs.get(int(msg.channel), program)),
                    )
                )
        # Close unterminated notes at the track end.
        end_tick = tick
        for (channel, pitch), queue in active.items():
            while queue:
                start_tick, velocity = queue.popleft()
                if channel == 9:
                    continue
                notes.append(
                    MidiNote(
                        start=converter.seconds(start_tick),
                        end=converter.seconds(max(start_tick + 1, end_tick)),
                        pitch=pitch,
                        velocity=velocity,
                        channel=channel,
                        track=track_index,
                        program=channel_programs.get(channel, program),
                    )
                )
        tracks_info.append(
            {
                "index": track_index,
                "name": track.name,
                "note_count": track_notes,
                "programs": sorted(set(channel_programs.values())),
            }
        )

    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes, tracks_info, sorted(programs), signature, raw


def _estimate_key(notes: list[MidiNote]) -> tuple[int, str, float, dict[str, float]]:
    weights = np.zeros(12, dtype=np.float64)
    for note in notes:
        weights[note.pitch % 12] += max(0.01, note.duration) * (0.4 + note.velocity / 127.0)
    if weights.sum() <= 0:
        weights[:] = 1.0
    weights /= weights.sum()

    best: tuple[float, int, str] | None = None
    correlations: dict[str, float] = {}
    for tonic in range(12):
        major = np.corrcoef(weights, np.roll(_MAJOR_PROFILE, tonic))[0, 1]
        minor = np.corrcoef(weights, np.roll(_MINOR_PROFILE, tonic))[0, 1]
        correlations[f"{PC_NAMES[tonic]} major"] = float(major)
        correlations[f"{PC_NAMES[tonic]} minor"] = float(minor)
        for score, mode in ((major, "major"), (minor, "minor")):
            if best is None or score > best[0]:
                best = (float(score), tonic, mode)
    assert best is not None
    ordered = sorted(correlations.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    confidence = _clamp(0.5 * best[0] + 0.5 * _sigmoid(margin * 8.0), 0.0, 1.0)
    return best[1], best[2], confidence, correlations


def _degree_states(tonic: int, mode: str) -> list[tuple[int, str, tuple[int, ...], int]]:
    """Return (root, quality, intervals, degree) states for the key."""

    # For minor keys, use the relative-major degrees and mark the tonic degree.
    # The transcription is major, but this keeps the code honest for other files.
    base = tonic if mode == "major" else (tonic + 3) % 12
    states = []
    for degree, (offset, quality, intervals) in enumerate(_MAJOR_DEGREES):
        states.append(((base + offset) % 12, quality, intervals, degree))
    return states


def _estimate_chords(
    notes: list[MidiNote],
    tonic: int,
    mode: str,
    bpm: float,
    total_beats: float,
    harmonic_rhythm: float = 2.0,
    context_beats: float = 0.5,
) -> list[ChordEvent]:
    states = _degree_states(tonic, mode)
    window = max(0.5, float(harmonic_rhythm))
    n = max(1, int(math.ceil(total_beats / window)))
    spb = 60.0 / max(1.0, bpm)
    emission = np.zeros((n, len(states)), dtype=np.float64)
    bass_pcs: list[int | None] = []

    for wi in range(n):
        start = wi * window
        end = start + window
        lo = max(0.0, start - context_beats)
        hi = end + context_beats
        strong = np.zeros(12, dtype=np.float64)
        bass_pitch: int | None = None
        for note in notes:
            a = note.start / spb
            b = note.end / spb
            overlap = max(0.0, min(b, hi) - max(a, lo))
            if overlap <= 0:
                continue
            beat_in_bar = a % 4.0
            if beat_in_bar < 0.2 or abs(beat_in_bar - 2.0) < 0.2:
                metrical = 2.5
            elif abs(beat_in_bar - 1.0) < 0.2 or abs(beat_in_bar - 3.0) < 0.2:
                metrical = 1.2
            else:
                metrical = 0.35
            strong[note.pitch % 12] += overlap * metrical
            if bass_pitch is None or note.pitch < bass_pitch:
                bass_pitch = note.pitch
        bass_pcs.append(bass_pitch % 12 if bass_pitch is not None else None)
        total = max(1e-9, float(strong.sum()))
        for si, (root, quality, intervals, _degree) in enumerate(states):
            tones = [(root + i) % 12 for i in intervals]
            score = sum(strong[t] for t in tones) / len(tones) / total
            score -= 0.30 * sum(strong[j] for j in range(12) if j not in tones) / total
            if bass_pitch is not None:
                if bass_pitch % 12 == root:
                    score += 0.24
                elif bass_pitch % 12 in ((root + 7) % 12, (root + 4) % 12, (root + 3) % 12):
                    score += 0.05
            emission[wi, si] = score

    # Viterbi over the functional progression.
    dp = np.zeros((n, len(states)), dtype=np.float64)
    back = np.zeros((n, len(states)), dtype=np.int64)
    tonic_degree = 0
    dp[0] = emission[0] + 0.06 * np.array(
        [0.5 if d == tonic_degree else 0.0 for _r, _q, _i, d in states]
    )
    for wi in range(1, n):
        for j in range(len(states)):
            values = dp[wi - 1] + _DEGREE_TRANSITIONS[:, j]
            k = int(np.argmax(values))
            dp[wi, j] = values[k] + emission[wi, j]
            back[wi, j] = k
    path = [int(np.argmax(dp[-1]))]
    for wi in range(n - 1, 0, -1):
        path.append(int(back[wi, path[-1]]))
    path = path[::-1]

    # Segment and compute confidence from the emission margin.
    segments: list[ChordEvent] = []
    run_start = 0
    for wi in range(1, n + 1):
        if wi < n and path[wi] == path[run_start]:
            continue
        si = path[run_start]
        root, quality, _intervals, degree = states[si]
        row = np.sort(emission[run_start:wi, si])
        margin = float(row[-1] - np.median(emission[run_start:wi, si]))
        confidence = _clamp(_sigmoid(margin * 10.0), 0.0, 1.0)
        # Bass for the segment: most common bass pitch class.
        counts: dict[int, int] = {}
        for pc in bass_pcs[run_start:wi]:
            if pc is not None:
                counts[pc] = counts.get(pc, 0) + 1
        bass_pc = max(counts, key=counts.get) if counts else root
        bass_pitch = _pitch_for_pc(bass_pc, center=40)
        segments.append(
            ChordEvent(
                start_beat=run_start * window,
                end_beat=wi * window,
                root=root,
                quality=quality,
                degree=degree,
                confidence=confidence,
                bass_pitch=bass_pitch,
            )
        )
        run_start = wi

    # Merge very short segments into the musically closer neighbour.
    min_beats = max(1.0, window)
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, seg in enumerate(segments):
            if seg.end_beat - seg.start_beat >= min_beats:
                continue
            if i == 0:
                neighbour = segments[1]
            elif i == len(segments) - 1:
                neighbour = segments[-2]
            else:
                prev, nxt = segments[i - 1], segments[i + 1]
                neighbour = prev if prev.confidence >= nxt.confidence else nxt
            replacement = dataclasses.replace(
                neighbour,
                start_beat=min(seg.start_beat, neighbour.start_beat),
                end_beat=max(seg.end_beat, neighbour.end_beat),
            )
            if i > 0 and segments[i - 1] is neighbour:
                segments[i - 1] = replacement
                del segments[i]
            elif i + 1 < len(segments) and segments[i + 1] is neighbour:
                segments[i + 1] = replacement
                del segments[i]
            else:  # pragma: no cover - defensive
                segments[i] = replacement
            changed = True
            break
    # Collapse adjacent identical chords.
    merged: list[ChordEvent] = []
    for seg in segments:
        if merged and merged[-1].root == seg.root and merged[-1].quality == seg.quality:
            merged[-1] = dataclasses.replace(merged[-1], end_beat=seg.end_beat)
        else:
            merged.append(seg)
    return merged


def _pitch_for_pc(pc: int, center: int = 60) -> int:
    lo = max(0, int(center) - 18)
    hi = min(127, int(center) + 18)
    candidates = [p for p in range(lo, hi + 1) if p % 12 == int(pc) % 12]
    if not candidates:
        return int(center)
    return int(min(candidates, key=lambda p: abs(p - center)))


def _build_bars(
    notes: list[MidiNote],
    chords: list[ChordEvent],
    bpm: float,
    total_beats: float,
    beats_per_bar: float,
) -> list[BarInfo]:
    spb = 60.0 / max(1.0, bpm)
    n_bars = max(1, int(math.ceil(total_beats / beats_per_bar)))
    bars: list[BarInfo] = []
    for index in range(n_bars):
        start_beat = index * beats_per_bar
        end_beat = start_beat + beats_per_bar
        start_s = start_beat * spb
        end_s = end_beat * spb
        in_bar = [n for n in notes if start_s <= n.start < end_s]
        bass = min((n.pitch for n in in_bar), default=None)
        melody = max((n.pitch for n in in_bar), default=None)
        chord = next(
            (c for c in chords if c.start_beat <= start_beat < c.end_beat),
            chords[0] if chords else None,
        )
        bars.append(
            BarInfo(
                index=index,
                start_beat=start_beat,
                end_beat=end_beat,
                chord=chord.symbol if chord else "",
                bass_pitch=bass,
                melody_pitch=melody,
                note_count=len(in_bar),
                density_hz=len(in_bar) / max(1e-6, beats_per_bar * spb),
            )
        )
    return bars


def _select_loop(
    bars: list[BarInfo],
    chords: list[ChordEvent],
    loop_bars: int,
    beats_per_bar: float,
) -> int:
    if len(bars) <= loop_bars + 4:
        return 0
    # Skip the sparse intro/outro.
    first = min(8, max(0, len(bars) // 8))
    last = max(first + 1, len(bars) - loop_bars - 2)
    best_score = -1e9
    best_start = first
    for start in range(first, last):
        window = bars[start : start + loop_bars]
        if not window:
            continue
        density = float(np.mean([b.density_hz for b in window]))
        if density <= 0:
            continue
        bass_ok = 0
        changes = 0
        prev = None
        for bar in window:
            chord = next(
                (c for c in chords if c.start_beat <= bar.start_beat < c.end_beat),
                None,
            )
            if chord is not None and bar.bass_pitch is not None:
                if bar.bass_pitch % 12 in chord.pitch_classes:
                    bass_ok += 1
            if prev is not None and bar.chord != prev:
                changes += 1
            prev = bar.chord
        score = 0.015 * bass_ok + 0.04 * min(changes, 12) + 0.25 * min(density, 6.0)
        # Prefer a bit of harmonic motion but not chaos.
        score -= 0.05 * abs(changes - 9)
        if score > best_score:
            best_score = score
            best_start = start
    return int(best_start)


def _extract_theme(
    notes: list[MidiNote],
    loop_start_beat: float,
    loop_end_beat: float,
    bpm: float,
    beats_per_bar: float,
) -> list[ThemeNote]:
    """Clean upper voice: 8th-note grid, continuity-smoothed, deduplicated."""

    spb = 60.0 / max(1.0, bpm)
    step = 0.5
    start_s = loop_start_beat * spb
    end_s = loop_end_beat * spb
    candidates: dict[int, list[MidiNote]] = {}
    for note in notes:
        if not (55 <= note.pitch <= 88):
            continue
        if note.velocity < 30 or note.duration < 0.08:
            continue
        if note.start < start_s or note.start >= end_s:
            continue
        cell = int(math.floor((note.start - start_s) / (step * spb) + 1e-6))
        candidates.setdefault(cell, []).append(note)

    theme: list[ThemeNote] = []
    previous: int | None = None
    grid_beat = loop_start_beat
    last_cell = int(math.floor((end_s - start_s) / (step * spb)))
    for cell in range(last_cell):
        pool = candidates.get(cell, [])
        if not pool:
            continue
        if previous is None:
            chosen = max(pool, key=lambda n: (n.pitch, n.velocity))
        else:
            # Prefer a stepwise continuation but never lose the upper voice.
            chosen = min(
                pool,
                key=lambda n: abs(n.pitch - previous) - 0.10 * (n.pitch - 55),
            )
        beat = loop_start_beat + cell * step
        duration = _clamp((chosen.duration / spb) / step, 0.5, 4.0) * step
        if (
            theme
            and theme[-1].pitch == chosen.pitch
            and beat - (theme[-1].beat + theme[-1].duration_beats) <= step + 1e-6
        ):
            theme[-1] = dataclasses.replace(
                theme[-1],
                duration_beats=max(
                    theme[-1].duration_beats,
                    beat + duration - theme[-1].beat,
                ),
            )
        else:
            theme.append(
                ThemeNote(
                    beat=beat,
                    duration_beats=duration,
                    pitch=int(chosen.pitch),
                    velocity=int(_clamp(chosen.velocity, 45, 100)),
                )
            )
        previous = chosen.pitch
        grid_beat += step
    return theme


def analyze_midi(
    path: str | Path,
    *,
    loop_bars: int = 16,
    harmonic_rhythm_beats: float = 2.0,
    min_duration_seconds: float = 0.05,
    min_velocity: int = 15,
) -> MidiAnalysis:
    """Analyse a transcription and return deterministic, serialisable material."""

    import mido

    path = Path(path)
    midi = mido.MidiFile(str(path))
    tempo_changes = _tempo_map(midi)
    end_tick = max(
        (sum(int(msg.time) for msg in track) for track in midi.tracks),
        default=int(midi.ticks_per_beat) * 4,
    )
    converter = _TempoConverter(tempo_changes, int(midi.ticks_per_beat), end_tick)
    notes, tracks, programs, signature, raw_count = _parse_notes(midi, converter)

    # Basic Pitch writes a very long single tempo; use the first tempo event.
    bpm = 60_000_000.0 / tempo_changes[0][1]
    duration_seconds = float(midi.length)
    duration_beats = converter.beats(duration_seconds)
    beats_per_bar = signature[0] * 4.0 / signature[1]

    filtered = [
        n
        for n in notes
        if n.duration >= min_duration_seconds and n.velocity >= min_velocity
    ]
    if not filtered:
        filtered = notes

    tonic, mode, key_confidence, correlations = _estimate_key(filtered)
    chords = _estimate_chords(
        filtered,
        tonic,
        mode,
        bpm,
        duration_beats,
        harmonic_rhythm=harmonic_rhythm_beats,
    )
    bars = _build_bars(filtered, chords, bpm, duration_beats, beats_per_bar)
    loop_start_bar = _select_loop(bars, chords, loop_bars, beats_per_bar)
    loop_start_beat = loop_start_bar * beats_per_bar
    loop_end_beat = min(duration_beats, loop_start_beat + loop_bars * beats_per_bar)
    theme = _extract_theme(
        filtered, loop_start_beat, loop_end_beat, bpm, beats_per_bar
    )

    # Transcription diagnostics.  These are deliberately descriptive rather
    # than silently "corrected": the user can see what was unreliable.
    durations = np.array([n.duration for n in notes], dtype=np.float64)
    pitches = np.array([n.pitch for n in notes], dtype=np.int64)
    low_velocity = sum(1 for n in notes if n.velocity < 35)
    high_register = sum(1 for n in notes if n.pitch > 84)
    diatonic = {0, 2, 4, 5, 7, 9, 11}
    non_diatonic = sum(
        1 for n in notes if (n.pitch - tonic) % 12 not in diatonic
    )
    issues = {
        "single_track": len(tracks) <= 1,
        "tracks": len(tracks),
        "programs": programs,
        "vocal_instrumental_split": False,
        "note": (
            "Basic Pitch merges vocals and accompaniment into one track; no "
            "separate vocal/instrumental tracks exist. The upper voice is a "
            "best-effort melodic reduction, not a verified vocal stem."
        ),
        "short_notes": int(np.sum(durations < 0.08)),
        "low_velocity_notes": int(low_velocity),
        "high_register_notes": int(high_register),
        "non_diatonic_notes": int(non_diatonic),
        "raw_note_count": int(raw_count),
        "kept_note_count": len(filtered),
        "dropped_note_count": int(raw_count - len(filtered)),
    }
    stats = {
        "note_count": len(filtered),
        "onsets_per_second": round(len(filtered) / max(1e-6, duration_seconds), 3),
        "pitch_min": int(pitches.min()) if len(pitches) else 0,
        "pitch_median": int(np.median(pitches)) if len(pitches) else 0,
        "pitch_max": int(pitches.max()) if len(pitches) else 0,
        "duration_median_s": round(float(np.median(durations)), 4)
        if len(durations)
        else 0.0,
        "duration_mean_s": round(float(np.mean(durations)), 4)
        if len(durations)
        else 0.0,
        "pitch_class_weights": {
            PC_NAMES[i]: round(float(w), 4)
            for i, w in enumerate(
                np.bincount(
                    (pitches % 12) if len(pitches) else np.zeros(1, dtype=int),
                    minlength=12,
                ).astype(float)
                / max(1, len(pitches))
            )
        },
        "chord_count": len(chords),
        "chord_changes_per_bar": round(
            sum(1 for a, b in zip(chords, chords[1:]) if a.root != b.root)
            / max(1, len(bars)),
            3,
        ),
        "theme_notes": len(theme),
        "loop_note_count": sum(
            1
            for n in filtered
            if loop_start_beat * 60.0 / bpm
            <= n.start
            < loop_end_beat * 60.0 / bpm
        ),
    }

    return MidiAnalysis(
        source=str(path),
        duration_seconds=duration_seconds,
        duration_beats=duration_beats,
        bpm=bpm,
        ticks_per_beat=int(midi.ticks_per_beat),
        time_signature=signature,
        beats_per_bar=beats_per_bar,
        key_tonic=tonic,
        key_mode=mode,
        key_confidence=key_confidence,
        key_correlations=correlations,
        tracks=tracks,
        programs=programs,
        raw_note_count=raw_count,
        notes=filtered,
        chords=chords,
        bars=bars,
        theme=theme,
        loop_start_bar=loop_start_bar,
        loop_bars=loop_bars,
        loop_start_beat=loop_start_beat,
        loop_end_beat=loop_end_beat,
        issues=issues,
        stats=stats,
    )
