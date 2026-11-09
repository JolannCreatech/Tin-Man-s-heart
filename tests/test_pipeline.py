"""End-to-end pipeline test with the mock engine (no MRT2 load)."""

from __future__ import annotations

import json
import threading
import time

import soundfile as sf

from diastole.config import AudioConfig, RuntimeConfig
from diastole.runtime.pipeline import Pipeline
from diastole.sensors.base import StaticSensor


def test_pipeline_mock_session(tmp_path) -> None:
    config = RuntimeConfig(
        audio=AudioConfig(
            enabled=False, buffer_seconds=0.4, start_threshold_seconds=0.2
        ),
        duration=1.2,
        seed=3,
    )
    pipeline = Pipeline(
        config,
        mock=True,
        use_reference=False,
        sensor=StaticSensor(60),
        session_name="pipeline_test",
        output_dir=tmp_path,
    )
    report = pipeline.run(duration=1.2)

    assert report["session"]["frames"] == 30
    assert report["validation"]["continuous_frames"] is True
    assert report["validation"]["non_silent"] is True
    assert report["validation"]["midi_written"] is True
    assert report["validation"]["audible_output"] is False
    assert report["audio"]["frames"] == 30 * 1920
    assert report["timeline"]["note_events"] > 0
    assert report["latency_ms"]["frame_budget"] == 40.0
    assert report["latency_ms"]["count"] == 30

    assert pipeline.paths.wav.exists()
    assert pipeline.paths.midi.exists()
    assert pipeline.paths.report_json.exists()
    assert pipeline.paths.report_markdown.exists()
    data, sr = sf.read(pipeline.paths.wav, dtype="float32", always_2d=True)
    assert data.shape == (30 * 1920, 2)
    assert sr == 48000
    on_disk = json.loads(pipeline.paths.report_json.read_text())
    assert on_disk["session"]["name"] == "pipeline_test"


def test_concurrent_stop_writes_report_once(tmp_path) -> None:
    config = RuntimeConfig(
        audio=AudioConfig(
            enabled=False, buffer_seconds=0.4, start_threshold_seconds=0.2
        ),
        duration=5.0,
        seed=5,
    )
    pipeline = Pipeline(
        config,
        mock=True,
        use_reference=False,
        sensor=StaticSensor(60),
        session_name="concurrent_stop",
        output_dir=tmp_path,
    )
    pipeline.start()
    time.sleep(0.4)
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def shutdown() -> None:
        barrier.wait()
        try:
            pipeline.stop()
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [threading.Thread(target=shutdown) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    report = pipeline.load_report()
    assert report["session"]["frames"] > 0
    assert pipeline.paths.report_json.exists()
    assert pipeline.paths.report_markdown.exists()


def test_pipeline_manual_bpm_changes_processing(tmp_path) -> None:
    config = RuntimeConfig(
        audio=AudioConfig(
            enabled=False, buffer_seconds=0.4, start_threshold_seconds=0.2
        ),
        duration=1.0,
        seed=4,
    )
    pipeline = Pipeline(
        config,
        mock=True,
        use_reference=False,
        sensor=StaticSensor(60),
        session_name="manual_test",
        output_dir=tmp_path,
    )
    pipeline.set_manual_bpm(150)
    report = pipeline.run(duration=1.0)
    assert report["bpm"]["max"] > 60
    assert report["musical"]["density_mean"] > 0.8
    assert report["musical"]["drum_fraction"] > 0.02
