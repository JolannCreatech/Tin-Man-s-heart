"""Pitch profile and musical timeline / MIDI / scheduler tests."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest

from diastole.config import SchedulerConfig
from diastole.music.pitch import PitchProfile
from diastole.music.timeline import MusicalTimeline, TimelineScheduler
from diastole.signal.mapping import map_bpm


def write_test_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    for note in (60, 64, 67):
        track.append(mido.Message("note_on", note=note, velocity=90, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)


def test_pitch_profile_from_midi(tmp_path: Path) -> None:
    path = tmp_path / "test.mid"
    write_test_midi(path)
    profile = PitchProfile.from_midi(path, 48, 72)
    top = set(int(i) for i in profile.weights.argsort()[-3:])
    assert {0, 4, 7}.issubset(top)
    assert profile.source == str(path)
    chord = profile.chord(__import__("random").Random(1), count=3)
    assert len(chord) == 3
    assert all(48 <= p <= 72 for p in chord)


def test_pitch_profile_missing_falls_back(tmp_path: Path) -> None:
    profile = PitchProfile.from_midi(tmp_path / "missing.mid", 48, 72)
    assert profile.source == "fallback"
    assert profile.weights.sum() == pytest.approx(1.0)


def test_timeline_ticks_follow_tempo() -> None:
    timeline = MusicalTimeline(ppq=480)
    timeline.set_tempo(0.0, 120.0)
    assert timeline.seconds_to_ticks(1.0) == 960
    timeline.set_tempo(1.0, 60.0)
    # 1 s at 120 BPM = 960 ticks; 1 s at 60 BPM = 480 ticks.
    assert timeline.seconds_to_ticks(2.0) == 1440


def test_timeline_queries() -> None:
    timeline = MusicalTimeline()
    timeline.add_note(1.0, 60, 0.5)
    timeline.add_note(1.1, 64, 0.2)
    timeline.add_drum(1.2, 36)
    assert [n.pitch for n in timeline.notes_active_at(1.05)] == [60]
    assert [n.pitch for n in timeline.note_onsets_between(1.0, 1.2)] == [60, 64]
    assert len(timeline.drums_between(1.0, 1.3)) == 1
    assert timeline.end_time() == pytest.approx(1.5)


def test_midi_roundtrip(tmp_path: Path) -> None:
    timeline = MusicalTimeline()
    timeline.set_tempo(0.0, 100.0)
    timeline.add_note(0.0, 60, 0.5, velocity=88)
    timeline.add_note(0.5, 64, 0.5, velocity=80)
    timeline.add_drum(0.5, 36, velocity=100)
    path = timeline.write_midi(tmp_path / "score.mid")

    midi = mido.MidiFile(path)
    messages = [msg for track in midi.tracks for msg in track]
    tempos = [m for m in messages if m.type == "set_tempo"]
    ons = [m for m in messages if m.type == "note_on" and m.velocity > 0]
    offs = [m for m in messages if m.type == "note_off"]
    assert len(tempos) == 1
    assert len(ons) == 3
    assert len(offs) == 3
    assert any(m.channel == 9 and m.note == 36 for m in ons)
    assert {m.note for m in ons if m.channel == 0} == {60, 64}


def test_scheduler_lookahead_and_tempo_change() -> None:
    timeline = MusicalTimeline()
    scheduler = TimelineScheduler(SchedulerConfig(lookahead_seconds=1.0), seed=1)
    params = map_bpm(140)
    scheduler.update(0.0, params, timeline)
    assert timeline.note_count + timeline.drum_count > 0
    assert timeline.tempo_changes
    first_end = timeline.end_time()
    assert first_end <= 1.0 + 60 / params.tempo_bpm

    # Advance and change tempo: events continue forward, not backwards.
    scheduler.update(10.0, map_bpm(60), timeline)
    assert timeline.note_count + timeline.drum_count > 0
    assert timeline.end_time() >= first_end
    tempos = timeline.tempo_changes
    assert len(tempos) >= 2


def test_scheduler_handles_stall_without_past_burst() -> None:
    timeline = MusicalTimeline()
    scheduler = TimelineScheduler(SchedulerConfig(lookahead_seconds=0.2), seed=2)
    params = map_bpm(120)
    scheduler.update(0.0, params, timeline)
    before = timeline.note_count + timeline.drum_count
    # Jump far ahead: only the future lookahead window should be scheduled.
    scheduler.update(30.0, params, timeline)
    added = timeline.note_count + timeline.drum_count - before
    assert added > 0
    assert timeline.end_time() <= 30.0 + 0.2 + 60 / params.tempo_bpm
