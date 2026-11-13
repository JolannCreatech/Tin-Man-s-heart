"""Tests for the iteration-2 musical layers.

These are deliberately model-free: MIDI analysis, offline rendering, the
BPM-locked arranger, the real-time synths and the mixer all run on CPU.
"""

from __future__ import annotations

import numpy as np

from diastole.audio.mixer import LayerMixer, Limiter, SmoothedGain
from diastole.audio.synth import AccompanimentSynth, HeartbeatSynth
from diastole.config import REFERENCE_MIDI, ArrangementConfig, MixConfig
from diastole.music.arranger import MidiArranger
from diastole.music.midi_analysis import analyze_midi
from diastole.music.midi_render import (
    load_or_render,
    loop_material,
    render_arrangement,
    render_voice,
)


def test_midi_analysis_localises_song() -> None:
    analysis = analyze_midi(REFERENCE_MIDI)
    assert analysis.key_name == "Eb major"
    assert abs(analysis.bpm - 120.0) < 1e-6
    assert analysis.duration_beats > 300
    assert analysis.loop_bars == 16
    assert analysis.loop_start_beat >= 0
    assert len(analysis.chords) > 20
    assert len(analysis.theme) > 10
    # The selected loop must actually fall inside the song.
    assert analysis.loop_end_beat <= analysis.duration_beats + 1e-6
    assert 0.0 <= analysis.key_confidence <= 1.0


def test_loop_material_and_render_are_deterministic(tmp_path) -> None:
    analysis = analyze_midi(REFERENCE_MIDI)
    material = loop_material(analysis)
    assert material.loop_length_beats == 64.0
    assert material.chords
    assert material.theme

    kwargs = dict(seconds=4.0, sample_rate=8000, seed=3)
    a = render_arrangement(analysis, **kwargs)
    b = render_arrangement(analysis, **kwargs)
    assert a.shape == (4 * 8000, 2)
    assert a.dtype == np.float32
    assert np.max(np.abs(a)) > 0.1
    assert np.max(np.abs(a)) <= 1.0
    assert np.array_equal(a, b)

    audio, path = load_or_render(
        analysis,
        seconds=2.0,
        sample_rate=8000,
        directory=tmp_path,
    )
    assert path.exists()
    assert audio.shape == (2 * 8000, 2)


def test_render_voice_shapes_and_decay() -> None:
    for kind in ("bass", "pad", "lead"):
        wave = render_voice(kind, 60, 0.25, 0.8, 8000)
        assert wave.dtype == np.float32
        assert wave.shape == (2000,)
        assert np.max(np.abs(wave)) > 0.0


def test_arranger_is_beat_locked() -> None:
    analysis = analyze_midi(REFERENCE_MIDI)
    arranger = MidiArranger(analysis, ArrangementConfig(mode="hybrid"), sample_rate=8000)
    assert arranger.loop_beats == 64.0

    # 4 s at 60 BPM advances exactly 4 beats; 4 s at 120 BPM advances 8.
    for bpm, expected in ((60.0, 4.0), (120.0, 8.0)):
        arranger.reset()
        events = 0
        for frame in range(100):
            batch, _ = arranger.update(frame * 0.04, bpm)
            events += len(batch)
        assert abs(arranger.beat - expected) < 1e-6
        assert events > 0

    # Events carry sample-accurate starts and positive durations.
    arranger.reset()
    batch, drums = arranger.update(0.0, 60.0)
    for event in batch:
        assert event.start_sample >= 0
        assert event.duration > 0.0
        assert 0 < event.velocity <= 127
    assert drums  # heartbeat documentation exists


def test_heartbeat_stays_bpm_exact_and_independent() -> None:
    for bpm, seconds, expected in ((60.0, 2.0, 2), (120.0, 2.0, 4)):
        synth = HeartbeatSynth(8000, ArrangementConfig(heartbeat=True))
        synth.set_bpm(bpm)
        total = 0
        for start in range(0, 2 * 8000, 800):
            block = synth.render(start, 800)
            assert block.shape == (800,)
            total += int(np.count_nonzero(block))
        assert synth.beat_events == expected
        assert total > 0

    silent = HeartbeatSynth(8000, ArrangementConfig(heartbeat=False))
    silent.set_bpm(120.0)
    assert not np.any(silent.render(0, 8000))
    assert silent.beat_events == 0


def test_accompaniment_synth_schedules_voices() -> None:
    synth = AccompanimentSynth(8000)
    synth.add(0, "bass", 48, 0.5, 0.9)
    first = synth.render(0, 1000)
    assert np.count_nonzero(first) > 0
    # The tail still has energy after the first block.
    assert np.count_nonzero(synth.render(1000, 1000)) > 0
    synth.clear()
    assert synth.pending == 0


def test_mixer_gains_limiter_and_reference() -> None:
    config = MixConfig(
        heartbeat_gain=0.5,
        accompaniment_gain=0.0,
        mrt2_gain=0.5,
        master_gain=1.0,
        reference_gain=0.0,
        ducking=0.0,
    )
    mixer = LayerMixer(config, sample_rate=8000)
    frames = 800
    mrt2 = np.full((frames, 2), 0.3, dtype=np.float32)
    heart = np.full(frames, 0.3, dtype=np.float32)
    accomp = np.full(frames, 0.3, dtype=np.float32)
    out = mixer.process(mrt2, heart, accomp)
    assert out.shape == (frames, 2)
    assert np.all(np.abs(out) <= 1.0)
    # 0.3*0.5 + 0.3*0.5, both smoothed from their targets.
    assert abs(float(out[-1, 0]) - 0.3) < 0.02

    # A loud sum must be limited rather than clipped abruptly.
    loud = mixer.process(
        np.full((frames, 2), 2.0, dtype=np.float32),
        np.zeros(frames, dtype=np.float32),
        np.zeros(frames, dtype=np.float32),
    )
    assert np.all(np.abs(loud) <= 1.0)
    assert mixer.limiter.reductions > 0

    ref = np.full((frames, 2), 0.2, dtype=np.float32)
    mixer.set_reference(ref)
    out = mixer.process(mrt2, np.zeros(frames), np.zeros(frames))
    assert out.shape == (frames, 2)


def test_smoothed_gain_approaches_target() -> None:
    gain = SmoothedGain(0.0, sample_rate=8000, seconds=0.05)
    gain.set(1.0)
    values = gain.process(2000)
    assert values[0] < 0.2
    assert values[-1] > 0.9
    assert np.all(np.diff(values) >= 0)


def test_limiter_preserves_quiet_signal() -> None:
    limiter = Limiter(0.9, sample_rate=8000, release_seconds=0.2)
    x = np.full((100, 2), 0.1, dtype=np.float32)
    out = limiter.process(x)
    assert np.allclose(out, x)
    assert limiter.reductions == 0
