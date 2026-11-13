"""Reference ingestion and cache-index tests (no model download needed)."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import pytest

from diastole.config import ReferenceConfig
from diastole.music.reference import ReferenceMusic, load_audio


def _make_reference(tmp_path):
    audio_path = tmp_path / "ref.wav"
    sr = 44100
    t = np.arange(2 * sr, dtype=np.float32) / sr
    sf.write(audio_path, 0.2 * np.sin(2 * np.pi * 220 * t), sr)
    midi_path = tmp_path / "ref.mid"
    import mido

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(midi_path)
    config = ReferenceConfig(
        audio_path=audio_path,
        midi_path=midi_path,
        cache_dir=tmp_path / "cache",
        style_clip_seconds=1.0,
        style_hop_seconds=0.5,
        style_speed=1.0,
        style_offset_seconds=0.0,
        prefill_excerpt_start=0.0,
        prefill_excerpt_seconds=1.0,
        conditioning_source="audio",
        style_mode="drift",
        text_style_blend=0.0,
    )
    return config


def test_load_audio_resamples_to_48k_stereo(tmp_path) -> None:
    config = _make_reference(tmp_path)
    data, sr = load_audio(config.audio_path, 48000)
    assert sr == 48000
    assert data.shape[1] == 2
    assert data.shape[0] == pytest.approx(96000, rel=0.01)
    assert np.max(np.abs(data)) <= 1.0


def test_reference_audio_and_pitch(tmp_path) -> None:
    config = _make_reference(tmp_path)
    ref = ReferenceMusic(config)
    audio = ref.load_audio()
    assert ref.duration == pytest.approx(2.0, rel=0.01)
    assert audio.shape[1] == 2
    profile = ref.pitch_profile()
    assert profile.source == str(config.midi_path)
    assert profile.weights[0] > 0


def test_style_window_indices_injected(tmp_path) -> None:
    config = _make_reference(tmp_path)
    ref = ReferenceMusic(config)
    ref.load_audio()
    # Inject a synthetic cache so this test never loads MusicCoCa.
    ref._style_tokens = np.arange(4 * 12, dtype=np.int32).reshape(4, 12)
    ref._style_starts = np.array([0.0, 0.5, 1.0, 1.5])
    assert ref.style_index_at(0.0) == 0
    assert ref.style_index_at(0.6) == 1
    assert ref.style_index_at(1.1) == 2
    # Session time cycles through the song.
    assert ref.style_index_at(ref.duration + 0.1) == 0
    window = ref.style_window_at(0.6)
    assert window["index"] == 1
    assert window["start_seconds"] == pytest.approx(0.5)
    assert window["tokens"] == list(range(12, 24))


def test_missing_reference_cache_falls_back(tmp_path) -> None:
    config = _make_reference(tmp_path)
    ref = ReferenceMusic(config)
    ref.load_audio()
    # No style cache prepared: the runtime fallback must stay well-formed.
    assert ref.style_tokens_for_session(0.0) == [-1] * 12
