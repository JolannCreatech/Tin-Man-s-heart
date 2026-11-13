"""Runtime orchestration: sensor -> processing -> mapping -> control -> MRT2 -> audio.

Threading model
---------------
* sensor thread     4 Hz, independent of inference
* generation thread up to 25 frames/s, owns the MRT2 state
* playback thread   blocking consumer of the ring buffer
* main/CLI thread   dashboard or command reader only

The generation thread never touches the audio device; the playback thread never
runs inference.  A pre-roll fills the ring buffer before playback starts so a
slow first inference does not produce an audible gap.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from diastole import config as cfg
from diastole.audio.mixer import RealtimeLayers
from diastole.audio.playback import AudioPlayer, RingBuffer, WavRecorder
from diastole.config import (
    FRAME_SECONDS,
    RuntimeConfig,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from diastole.generation.engine import MRT2Engine, MockEngine
from diastole.music.adapter import FrameAdapter
from diastole.music.arranger import MidiArranger
from diastole.music.midi_analysis import analyze_midi
from diastole.music.reference import ReferenceMusic
from diastole.music.timeline import MusicalTimeline, TimelineScheduler
from diastole.sensors.base import HeartRateSensor, PhysiologicalSample
from diastole.sensors.simulated import SimulatedHeartRateSensor
from diastole.signal.mapping import BpmMapper, MusicalParameters
from diastole.signal.processing import BpmProcessor, PhysiologicalState

Progress = Callable[[str], None] | None


def _round(value: float, digits: int = 3) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return round(float(value), digits)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class Metrics:
    """Thread-safe rolling metrics for the dashboard and the report."""

    def __init__(self, capacity: int = 200) -> None:
        self._lock = threading.RLock()
        self._latencies: deque[float] = deque(maxlen=capacity)
        self._all_latencies: list[float] = []
        self._frames: list[dict] = []
        self.latest: dict = {}
        self.started_at = time.time()
        self.stopped_at: float | None = None
        self.started_monotonic = time.monotonic()
        self.stopped_monotonic: float | None = None
        self.notes_scheduled = 0
        self.drums_scheduled = 0
        self.pre_roll_frames = 0
        self.dropped_frames = 0

    def record(
        self,
        *,
        audio_time: float,
        wall_time: float,
        state: PhysiologicalState,
        params: MusicalParameters,
        active_notes: int,
        onsets: int,
        drum: bool,
        latency_ms: float,
        buffer_fill: float,
        underruns: int,
        style_index: int,
    ) -> None:
        with self._lock:
            self._latencies.append(float(latency_ms))
            self._all_latencies.append(float(latency_ms))
            frame = {
                "t": _round(audio_time, 3),
                "wall": _round(wall_time, 3),
                "bpm_raw": _round(state.raw_bpm, 2),
                "bpm": _round(state.bpm, 2),
                "arousal": _round(state.arousal, 4),
                "trend": _round(state.trend_bpm_per_s, 3),
                "valid": bool(state.valid),
                "tempo": _round(params.tempo_bpm, 2),
                "energy": _round(params.energy, 4),
                "density": _round(params.rhythmic_density, 4),
                "note_activity": _round(params.note_activity, 4),
                "dynamics": _round(params.dynamics, 4),
                "intensity": _round(params.rhythmic_intensity, 4),
                "subdivision": int(params.beat_subdivision),
                "drum_gate": bool(params.drum_gate),
                "active_notes": int(active_notes),
                "onsets": int(onsets),
                "drum": int(bool(drum)),
                "gen_ms": _round(latency_ms, 2),
                "buffer": _round(buffer_fill, 3),
                "underruns": int(underruns),
                "style_index": int(style_index),
            }
            self._frames.append(frame)
            self.latest = frame

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self._latencies)
            return {
                "latest": dict(self.latest),
                "frames": len(self._frames),
                "notes_scheduled": self.notes_scheduled,
                "drums_scheduled": self.drums_scheduled,
                "pre_roll_frames": self.pre_roll_frames,
                "dropped_frames": self.dropped_frames,
                "latency_mean_ms": _round(float(np.mean(latencies)) if latencies else 0.0, 2),
                "latency_p95_ms": _round(_percentile(latencies, 95), 2),
                "elapsed": _round(time.monotonic() - self.started_monotonic, 2),
            }

    def frames(self) -> list[dict]:
        with self._lock:
            return list(self._frames)

    def latency_summary(self) -> dict:
        with self._lock:
            latencies = list(self._all_latencies)
        if not latencies:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
        return {
            "mean": _round(float(np.mean(latencies)), 2),
            "p50": _round(_percentile(latencies, 50), 2),
            "p95": _round(_percentile(latencies, 95), 2),
            "max": _round(float(np.max(latencies)), 2),
            "count": len(latencies),
        }

    def stop(self) -> None:
        with self._lock:
            self.stopped_at = time.time()
            self.stopped_monotonic = time.monotonic()


@dataclasses.dataclass
class SessionPaths:
    directory: Path
    wav: Path
    midi: Path
    report_json: Path
    report_markdown: Path


class Pipeline:
    """One continuous DIASTOLE session."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        engine: object | None = None,
        reference: ReferenceMusic | None = None,
        sensor: HeartRateSensor | None = None,
        session_name: str | None = None,
        output_dir: str | Path | None = None,
        mock: bool = False,
        use_reference: bool = True,
        progress: Progress = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.progress = progress or (lambda _msg: None)
        self.output_dir = Path(output_dir) if output_dir is not None else self.config.output_dir
        self.session_name = session_name or time.strftime("session_%Y%m%d_%H%M%S")
        self.paths = SessionPaths(
            directory=self.output_dir / self.session_name,
            wav=self.output_dir / self.session_name / "audio.wav",
            midi=self.output_dir / self.session_name / "score.mid",
            report_json=self.output_dir / self.session_name / "report.json",
            report_markdown=self.output_dir / self.session_name / "report.md",
        )
        self.paths.directory.mkdir(parents=True, exist_ok=True)

        self.mock = bool(mock)
        self.use_reference = bool(use_reference)
        self.reference = reference or ReferenceMusic(self.config.reference)
        self.engine = engine if engine is not None else (MockEngine() if mock else MRT2Engine(self.config.engine))
        self.sensor = sensor or SimulatedHeartRateSensor(
            scenario=self.config.scenario,
            seed=self.config.seed or 0,
        )

        self.processor = BpmProcessor(self.config.processor)
        self.mapper = BpmMapper(self.config.mapping)
        self.profile = self.reference.pitch_profile()
        self.scheduler = TimelineScheduler(
            self.config.scheduler, profile=self.profile, seed=self.config.seed
        )
        self.timeline = MusicalTimeline()
        self.cond_timeline = self.timeline
        self.adapter = FrameAdapter(mask_unused_notes=False)

        self.mode = self.config.arrangement.mode
        self.legacy = self.mode == "legacy"
        self.analysis = None
        self.arranger: MidiArranger | None = None
        self.layers: RealtimeLayers | None = None
        # The heartbeat is useful on its own, so build the layer stack whenever
        # it is enabled even if the melodic material still comes from the
        # iteration-1 scheduler (``legacy`` mode).
        if not self.legacy or self.config.arrangement.heartbeat:
            self.layers = RealtimeLayers(
                self.config.mix, self.config.arrangement, SAMPLE_RATE
            )

        self.ring = RingBuffer(self.config.audio.buffer_frames, self.config.audio.channels)
        self.recorder = WavRecorder()
        self.player = AudioPlayer(
            self.config.audio,
            self.ring,
            processor=self.layers.process if self.layers is not None else None,
            recorder=self.recorder,
        )
        self.metrics = Metrics()

        self._stop_event = threading.Event()
        self._gen_thread: threading.Thread | None = None
        self._audio_time = 0.0
        self._frames = 0
        self._style_prepared = False
        self._prefill_used = False
        self._prefill_seconds = 0.0
        self._prefill_load_seconds = 0.0
        self._prefill_cache_hit: bool | None = None
        self._prefill_cache = ""
        self._reference_error: str | None = None
        self._last_midi_write = -1.0
        self._started = False
        self._stopped = False
        self._stop_finished = threading.Event()
        self._lock = threading.Lock()

    # -- preparation ------------------------------------------------------
    def prepare(self) -> None:
        if not self.legacy:
            try:
                self.progress("Analysing MIDI transcription...")
                self.analysis = analyze_midi(
                    self.config.reference.midi_path,
                    loop_bars=self.config.arrangement.loop_bars,
                )
                self.analysis.write(self.paths.directory / "midi_analysis.json")
                self.reference.set_analysis(self.analysis)
                self.arranger = MidiArranger(
                    self.analysis, self.config.arrangement, sample_rate=SAMPLE_RATE
                )
                self.cond_timeline = MusicalTimeline()
                self.progress(
                    f"MIDI: {self.analysis.key_name}, loop bars "
                    f"{self.analysis.loop_start_bar}-"
                    f"{self.analysis.loop_start_bar + self.analysis.loop_bars}, "
                    f"{len(self.analysis.theme)} theme notes"
                )
            except Exception as exc:
                self.progress(
                    f"MIDI analysis failed, falling back to legacy: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.legacy = True
                if self.layers is not None and not self.config.arrangement.heartbeat:
                    self.layers = None
                    self.player.processor = None

        if self.use_reference:
            try:
                self.reference.prepare_style_tokens(progress=self.progress)
                self._style_prepared = True
                self.progress(f"Reference style windows: {len(self.reference.style_tokens)}")
            except Exception as exc:
                self._reference_error = f"{type(exc).__name__}: {exc}"
                self.progress(f"Reference style unavailable, using masked style: {exc}")

        if self.config.engine.prefill and not self.mock:
            try:
                state = self.reference.prefill_state(self.engine, progress=self.progress)
                self.engine.set_state(state)
                self._prefill_used = True
                self._prefill_seconds = float(
                    getattr(self.reference, "prefill_compute_seconds", 0.0)
                )
                self._prefill_load_seconds = float(
                    getattr(self.reference, "prefill_load_seconds", 0.0)
                )
                self._prefill_cache_hit = getattr(
                    self.reference, "prefill_cache_hit", None
                )
                self._prefill_cache = str(
                    self.reference.prefill_state_cache_path
                )
                self.progress("Prefill state installed; generation will continue the excerpt.")
            except Exception as exc:
                self.progress(f"Prefill unavailable, starting cold: {type(exc).__name__}: {exc}")

    def start(self) -> None:
        with self._lock:
            if self._started or self._stopped:
                return
            self._started = True
        self.paths.directory.mkdir(parents=True, exist_ok=True)
        self.prepare()
        self.recorder.open(self.paths.wav, SAMPLE_RATE, self.config.audio.channels)
        self.sensor.start()

        # The cold model's very first frame is silent (initial state transient).
        # Prime it once with fully masked conditioning and discard that frame.
        if not self.mock and not self._prefill_used:
            try:
                self.engine.step(
                    [-1] * cfg.STYLE_TOKEN_COUNT,
                    [-1] * cfg.NOTE_TOKEN_COUNT,
                    (-1,),
                )
            except Exception as exc:
                self.progress(f"Prime step failed: {exc}")

        # Pre-roll: fill the ring before the playback thread starts.
        target = min(
            self.ring.capacity,
            max(self.config.audio.start_threshold_frames, SAMPLES_PER_FRAME),
        )
        while self.ring.available < target and not self._stop_event.is_set():
            if self.config.duration is not None and self._audio_time >= self.config.duration:
                break
            self._generate_one(pre_roll=True)
        self.metrics.pre_roll_frames = self._frames
        self.progress(
            f"Pre-roll {self.ring.available / SAMPLE_RATE:.2f}s buffered "
            f"({self._frames} frames)"
        )

        self.player.start()
        self.progress(f"Audio sink: {self.player.mode}" + (f" ({self.player.error})" if self.player.error else ""))

        self._gen_thread = threading.Thread(
            target=self._generation_loop, name="diastole-generation", daemon=True
        )
        self._gen_thread.start()

    # -- generation -------------------------------------------------------
    def _style_tokens(self, audio_time: float) -> list[int]:
        if not self._style_prepared:
            return [-1] * cfg.STYLE_TOKEN_COUNT
        try:
            return self.reference.style_tokens_at(audio_time)
        except Exception:
            return [-1] * cfg.STYLE_TOKEN_COUNT

    def _generate_one(self, pre_roll: bool = False) -> bool:
        if self.ring.free < SAMPLES_PER_FRAME:
            return False
        wall_now = time.monotonic()
        sample = self.sensor.read()
        state = self.processor.update(sample, now=wall_now)
        params = self.mapper.map(state)
        audio_time = self._audio_time

        if self.legacy or self.arranger is None:
            added = self.scheduler.update(audio_time, params, self.timeline)
            cond_timeline = self.timeline
            if self.layers is not None:
                self.layers.set_bpm(params.tempo_bpm)
        else:
            self.timeline.set_tempo(audio_time, params.tempo_bpm)
            self.cond_timeline.set_tempo(audio_time, params.tempo_bpm)
            events, drums = self.arranger.update(audio_time, params.tempo_bpm)
            added = len(events) + len(drums)
            for event in events:
                velocity = int(min(127, max(1, event.velocity)))
                self.timeline.add_note(
                    event.time, event.pitch, event.duration, velocity=velocity
                )
                if event.kind in ("bass", "pad"):
                    self.cond_timeline.add_note(
                        event.time, event.pitch, event.duration, velocity=velocity
                    )
                if self.layers is not None:
                    self.layers.add_accompaniment(
                        event.start_sample,
                        event.kind,
                        event.pitch,
                        event.duration,
                        velocity / 127.0,
                    )
            for drum in drums:
                self.timeline.add_drum(drum.time, drum.note, drum.velocity)
            if self.layers is not None:
                self.layers.set_bpm(params.tempo_bpm)
            cond_timeline = self.cond_timeline

        if added:
            self.metrics.notes_scheduled = self.timeline.note_count
            self.metrics.drums_scheduled = self.timeline.drum_count

        style = self._style_tokens(audio_time)
        cond = self.adapter.conditioning(audio_time, cond_timeline, style)
        samples, latency_ms = self.engine.step(
            cond.style_tokens, cond.frame.notes, (cond.frame.drum,)
        )
        if samples.shape != (SAMPLES_PER_FRAME, self.config.audio.channels):
            raise RuntimeError(f"Engine returned {samples.shape}, expected {(SAMPLES_PER_FRAME, 2)}")

        written = self.ring.write(samples)
        if written < samples.shape[0]:
            self.metrics.dropped_frames += samples.shape[0] - written

        style_index = -1
        if self._style_prepared:
            try:
                style_index = self.reference.style_index_at(audio_time)
            except Exception:
                style_index = -1
        self.metrics.record(
            audio_time=audio_time,
            wall_time=wall_now - self.metrics.started_monotonic,
            state=state,
            params=params,
            active_notes=len(cond.frame.active_pitches),
            onsets=len(cond.frame.onset_pitches),
            drum=cond.frame.drum_hit,
            latency_ms=latency_ms,
            buffer_fill=self.ring.fill_fraction,
            underruns=self.player.underruns,
            style_index=style_index,
        )

        self._audio_time += FRAME_SECONDS
        self._frames += 1

        # Periodic atomic MIDI snapshot (one file, never one per measurement).
        if self._audio_time - self._last_midi_write >= 2.0:
            try:
                self.timeline.write_midi(self.paths.midi)
            except Exception as exc:
                self.progress(f"MIDI snapshot failed: {exc}")
            self._last_midi_write = self._audio_time
        return True

    def _generation_loop(self) -> None:
        target_fill = self.config.audio.target_fill_fraction
        try:
            while not self._stop_event.is_set():
                if self.config.duration is not None and self._audio_time >= self.config.duration:
                    break
                if self.ring.fill_fraction >= target_fill:
                    self._stop_event.wait(0.004)
                    continue
                if not self._generate_one():
                    self._stop_event.wait(0.002)
        except Exception as exc:  # pragma: no cover - surface into report
            self._reference_error = f"generation error: {type(exc).__name__}: {exc}"
            self.progress(self._reference_error)
        finally:
            self._stop_event.set()

    # -- control ----------------------------------------------------------
    def set_manual_bpm(self, bpm: float | None) -> None:
        self.sensor.set_manual_bpm(bpm)

    def set_scenario(self, scenario: str) -> None:
        if hasattr(self.sensor, "set_scenario"):
            self.sensor.set_scenario(scenario)  # type: ignore[attr-defined]

    def snapshot(self) -> dict:
        snap = self.metrics.snapshot()
        snap.update(
            {
                "audio_mode": self.player.mode,
                "audible": self.player.is_audible,
                "underruns": self.player.underruns,
                "buffer_seconds": round(self.ring.available / SAMPLE_RATE, 3),
                "buffer_fraction": round(self.ring.fill_fraction, 3),
                "notes_total": self.timeline.note_count,
                "drums_total": self.timeline.drum_count,
                "prefill": self._prefill_used,
                "reference_error": self._reference_error,
                "sensor_status": self.sensor.status.value,
                "mode": self.mode,
                "heartbeat_beat": (
                    round(self.layers.heartbeat.beat, 2)
                    if self.layers is not None
                    else None
                ),
            }
        )
        return snap

    def stop(self) -> None:
        with self._lock:
            first = not self._stopped
            if first:
                self._stopped = True
        if not first:
            # Another thread is already shutting down (e.g. interactive `stop`
            # racing the run loop). Wait until the report is on disk instead of
            # returning early and letting the caller read a missing report.
            self._stop_finished.wait(timeout=10.0)
            return

        try:
            self._stop_event.set()
            thread = self._gen_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)
            self._gen_thread = None
            self.player.stop()
            try:
                self.sensor.stop()
            finally:
                self.recorder.close()
            self.metrics.stop()

            try:
                self.timeline.write_midi(self.paths.midi)
            except Exception as exc:
                self.progress(f"Final MIDI write failed: {exc}")
            report = self.build_report()
            self.paths.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.paths.report_markdown.write_text(
                self._report_markdown(report), encoding="utf-8"
            )
            self.progress(f"Wrote {self.paths.wav}")
            self.progress(f"Wrote {self.paths.midi}")
            self.progress(f"Wrote {self.paths.report_json}")
        finally:
            self._stop_finished.set()

    def run(
        self,
        duration: float | None = None,
        dashboard: Callable[[dict], None] | None = None,
        dashboard_interval: float = 0.2,
    ) -> dict:
        if duration is not None:
            self.config = dataclasses.replace(self.config, duration=float(duration))
        self.start()
        try:
            if dashboard is None:
                thread = self._gen_thread
                if thread is not None:
                    while thread.is_alive() and not self._stop_event.is_set():
                        time.sleep(0.05)
            else:
                while not self._stop_event.is_set() and (
                    self._gen_thread is not None and self._gen_thread.is_alive()
                ):
                    dashboard(self.snapshot())
                    time.sleep(dashboard_interval)
        except KeyboardInterrupt:
            self.progress("Interrupted; shutting down gracefully...")
        finally:
            self.stop()
        return self.load_report()

    def load_report(self) -> dict:
        if self.paths.report_json.exists():
            return json.loads(self.paths.report_json.read_text(encoding="utf-8"))
        return {}

    # -- reporting --------------------------------------------------------
    def _audio_stats(self) -> dict:
        try:
            import soundfile as sf

            info = sf.info(str(self.paths.wav))
            data, sr = sf.read(str(self.paths.wav), dtype="float32", always_2d=True)
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            rms = float(np.sqrt(np.mean(data**2))) if data.size else 0.0
            clipped = int(np.sum(np.abs(data) >= 0.999)) if data.size else 0
            nan = bool(np.isnan(data).any())
            return {
                "exists": True,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "frames": int(info.frames),
                "seconds": round(info.frames / info.samplerate, 3),
                "peak": round(peak, 5),
                "rms": round(rms, 5),
                "clipped_samples": clipped,
                "clipped_fraction": round(clipped / data.size, 8) if data.size else 0.0,
                "nan": nan,
                "silent": peak < 0.01,
            }
        except Exception as exc:
            return {"exists": self.paths.wav.exists(), "error": str(exc)}

    def _arrangement_report(self) -> dict:
        from diastole.music.midi_render import summary as render_summary

        report: dict = {
            "mode": self.mode,
            "heartbeat": bool(self.config.arrangement.heartbeat),
            "loop_start_bar": None,
            "loop_bars": int(self.config.arrangement.loop_bars),
            "layers": self.layers.stats() if self.layers is not None else None,
        }
        if self.analysis is not None:
            report.update(
                {
                    "key": self.analysis.key_name,
                    "key_confidence": round(self.analysis.key_confidence, 4),
                    "bpm": round(self.analysis.bpm, 3),
                    "loop_start_bar": int(self.analysis.loop_start_bar),
                    "chords": [
                        {
                            "start_beat": round(c.start_beat, 2),
                            "end_beat": round(c.end_beat, 2),
                            "symbol": c.symbol,
                            "confidence": round(c.confidence, 3),
                        }
                        for c in self.analysis.chords
                        if c.end_beat > self.analysis.loop_start_beat
                        and c.start_beat
                        < self.analysis.loop_start_beat + self.analysis.loop_length_beats
                    ],
                    "theme_notes": len(self.analysis.theme),
                    "diagnostics": self.analysis.issues,
                    "stats": self.analysis.stats,
                    "render": render_summary(self.analysis),
                }
            )
        if self.arranger is not None:
            report["arranger"] = {
                "beat": round(self.arranger.beat, 3),
                "loop_beats": round(self.arranger.loop_beats, 3),
                "onsets": len(self.arranger._onsets),
            }
        return report

    def build_report(self) -> dict:
        frames = self.metrics.frames()
        latency = self.metrics.latency_summary()
        audio = self._audio_stats()

        def series(key: str) -> list[float]:
            return [float(f[key]) for f in frames if key in f]

        bpms = series("bpm")
        raws = series("bpm_raw")
        tempos = series("tempo")
        densities = series("density")
        activities = series("note_activity")
        active = series("active_notes")
        drums = series("drum")
        onsets = series("onsets")
        buffers = series("buffer")
        wall = series("wall")
        pre_roll = int(self.metrics.pre_roll_frames)
        startup_gaps = [round(b - a, 4) for a, b in zip(wall, wall[1:]) if b >= a]
        steady_wall = wall[pre_roll:] if pre_roll < len(wall) else []
        gaps = [round(b - a, 4) for a, b in zip(steady_wall, steady_wall[1:]) if b >= a]

        median_bpm = float(np.median(bpms)) if bpms else 60.0
        high_idx = [i for i, b in enumerate(bpms) if b >= median_bpm]
        low_idx = [i for i, b in enumerate(bpms) if b < median_bpm]
        audio_seconds = max(1e-9, self._audio_time)

        def mean_at(values: list[float], idx: list[int]) -> float:
            if not idx:
                return 0.0
            return _round(float(np.mean([values[i] for i in idx if i < len(values)])))

        def mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        onset_rate = mean(onsets) / FRAME_SECONDS
        drum_rate = mean(drums) / FRAME_SECONDS

        config_dict = self.config.as_dict()
        report = {
            "project": "DIASTOLE",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session": {
                "name": self.session_name,
                "started_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(self.metrics.started_at)
                ),
                "wall_seconds": _round(
                    (self.metrics.stopped_monotonic or time.monotonic())
                    - self.metrics.started_monotonic,
                    2,
                ),
                "audio_seconds": _round(self._audio_time, 3),
                "frames": self._frames,
                "mock": self.mock,
            },
            "config": config_dict,
            "engine": self.engine.info(),
            "reference": self.reference.info(),
            "arrangement": self._arrangement_report(),
            "prefill": {
                "used": self._prefill_used,
                "seconds": _round(self._prefill_seconds, 2),
                "compute_seconds": _round(self._prefill_seconds, 2),
                "load_seconds": _round(self._prefill_load_seconds, 3),
                "cache_hit": self._prefill_cache_hit,
                "cache": self._prefill_cache,
                "style_windows": (
                    int(self.reference.style_tokens.shape[0])
                    if self._style_prepared
                    else 0
                ),
                "error": self._reference_error,
            },
            "paths": {
                "directory": str(self.paths.directory),
                "wav": str(self.paths.wav),
                "midi": str(self.paths.midi),
                "report_json": str(self.paths.report_json),
                "report_markdown": str(self.paths.report_markdown),
            },
            "latency_ms": {
                **latency,
                "frame_budget": round(FRAME_SECONDS * 1000, 2),
                "real_time_factor": _round(
                    (latency["mean"] / (FRAME_SECONDS * 1000)) if latency["mean"] else 0.0
                ),
            },
            "audio": audio,
            "playback": self.player.stats(),
            "buffer": {
                "capacity_seconds": _round(self.ring.capacity / SAMPLE_RATE, 3),
                "max_fill_seconds": _round(max(buffers) * self.ring.capacity / SAMPLE_RATE, 3)
                if buffers
                else 0.0,
                "mean_fill_seconds": _round(
                    float(np.mean(buffers)) * self.ring.capacity / SAMPLE_RATE
                )
                if buffers
                else 0.0,
                "underruns": self.player.underruns,
                "pre_roll_frames": self.metrics.pre_roll_frames,
                "dropped_frames": self.metrics.dropped_frames,
            },
            "bpm": {
                "raw_min": _round(min(raws), 2) if raws else 0.0,
                "raw_max": _round(max(raws), 2) if raws else 0.0,
                "min": _round(min(bpms), 2) if bpms else 0.0,
                "max": _round(max(bpms), 2) if bpms else 0.0,
                "mean": _round(float(np.mean(bpms)), 2) if bpms else 0.0,
                "median": _round(median_bpm, 2),
            },
            "musical": {
                "tempo_min": _round(min(tempos), 2) if tempos else 0.0,
                "tempo_max": _round(max(tempos), 2) if tempos else 0.0,
                "tempo_mean": _round(float(np.mean(tempos)), 2) if tempos else 0.0,
                "density_mean": _round(float(np.mean(densities)), 4) if densities else 0.0,
                "note_activity_mean": _round(float(np.mean(activities)), 4) if activities else 0.0,
                "active_notes_mean": _round(float(np.mean(active)), 3) if active else 0.0,
                "onsets_mean": _round(float(np.mean(onsets)), 3) if onsets else 0.0,
                "drum_fraction": _round(float(np.mean(drums)), 3) if drums else 0.0,
                "onset_rate_hz": _round(onset_rate, 3),
                "drum_rate_hz": _round(drum_rate, 3),
                "note_events_per_s": _round(self.timeline.note_count / audio_seconds, 3),
                "drum_events_per_s": _round(self.timeline.drum_count / audio_seconds, 3),
                "high_bpm_active_notes": mean_at(active, high_idx),
                "low_bpm_active_notes": mean_at(active, low_idx),
                "high_bpm_drum_fraction": mean_at(drums, high_idx),
                "low_bpm_drum_fraction": mean_at(drums, low_idx),
                "high_bpm_density": mean_at(densities, high_idx),
                "low_bpm_density": mean_at(densities, low_idx),
                "high_bpm_onset_rate_hz": _round(mean_at(onsets, high_idx) / FRAME_SECONDS, 3),
                "low_bpm_onset_rate_hz": _round(mean_at(onsets, low_idx) / FRAME_SECONDS, 3),
                "high_bpm_drum_rate_hz": _round(mean_at(drums, high_idx) / FRAME_SECONDS, 3),
                "low_bpm_drum_rate_hz": _round(mean_at(drums, low_idx) / FRAME_SECONDS, 3),
                "segment_high_frames": len(high_idx),
                "segment_low_frames": len(low_idx),
            },
            "timeline": self.timeline.summary(),
            "generation_gaps_s": {
                "mean": _round(float(np.mean(gaps)), 4) if gaps else 0.0,
                "p95": _round(_percentile(gaps, 95), 4),
                "max": _round(max(gaps), 4) if gaps else 0.0,
                "startup_max": _round(max(startup_gaps), 4) if startup_gaps else 0.0,
                "pre_roll_frames": pre_roll,
                "note": "startup_max includes pre-roll/device-start handoff and is excluded from steady-state stats",
            },
            "validation": {
                "non_silent": bool(audio.get("exists") and not audio.get("silent", True)),
                "continuous_frames": self._frames == round(self._audio_time / FRAME_SECONDS),
                "midi_written": self.paths.midi.exists(),
                "prefilled": self._prefill_used,
                "audible_output": self.player.is_audible,
                "mode": self.mode,
                "heartbeat_layer": bool(
                    self.layers is not None and self.config.arrangement.heartbeat
                ),
                "mixed_layers": self.layers is not None,
            },
            "timeseries": frames,
        }
        return report

    @staticmethod
    def _report_markdown(report: dict) -> str:
        lat = report["latency_ms"]
        audio = report["audio"]
        bpm = report["bpm"]
        mus = report["musical"]
        val = report["validation"]
        pre = report["prefill"]
        return f"""# DIASTOLE session report

- Session: `{report['session']['name']}`
- Audio: **{report['session']['audio_seconds']} s**, {report['session']['frames']} frames
- Model: `{report['engine'].get('model_size')}` (mock: {report['session']['mock']})
- Reference style windows: {pre['style_windows']}; SpectroStream prefill: {pre['used']} ({pre['seconds']} s compute, {pre['load_seconds']} s cache load, cache hit: {pre['cache_hit']})
- Playback: {report['playback']['mode']} (audible: {report['playback']['audible']}, frames played: {report['playback']['frames_played']})

## Measured performance

| Metric | Value |
|---|---|
| Mean inference | {lat['mean']} ms/frame |
| p95 inference | {lat['p95']} ms/frame |
| Max inference | {lat['max']} ms/frame |
| Real-time budget | {lat['frame_budget']} ms/frame |
| Real-time factor | {lat['real_time_factor']} |
| Underruns | {report['buffer']['underruns']} |
| Dropped frames | {report['buffer']['dropped_frames']} |
| Generation gaps p95 | {report['generation_gaps_s']['p95']} s |
| Max startup/handoff gap | {report['generation_gaps_s']['startup_max']} s |

## Physiological response

| Metric | Value |
|---|---|
| BPM range (smoothed) | {bpm['min']} – {bpm['max']} |
| Tempo range | {mus['tempo_min']} – {mus['tempo_max']} |
| Mean active notes | {mus['active_notes_mean']} |
| Note onset rate | {mus['onset_rate_hz']} /s |
| Drum hit rate | {mus['drum_rate_hz']} /s |
| MIDI note events | {mus['note_events_per_s']} /s |
| MIDI drum events | {mus['drum_events_per_s']} /s |
| High-BPM active notes | {mus['high_bpm_active_notes']} |
| Low-BPM active notes | {mus['low_bpm_active_notes']} |
| High-BPM onset rate | {mus['high_bpm_onset_rate_hz']} /s |
| Low-BPM onset rate | {mus['low_bpm_onset_rate_hz']} /s |
| High-BPM drum rate | {mus['high_bpm_drum_rate_hz']} /s |
| Low-BPM drum rate | {mus['low_bpm_drum_rate_hz']} /s |
| High-BPM drum fraction | {mus['high_bpm_drum_fraction']} |
| Low-BPM drum fraction | {mus['low_bpm_drum_fraction']} |

## Audio validation

- Peak: {audio.get('peak')} / RMS: {audio.get('rms')} / silent: {audio.get('silent')}
- Clipped samples: {audio.get('clipped_samples')} ({audio.get('clipped_fraction')} of samples)
- Non-silent: **{val['non_silent']}**
- MIDI written: **{val['midi_written']}**
- Continuous frames: **{val['continuous_frames']}**

Files: `{report['paths']['wav']}`, `{report['paths']['midi']}`, `{report['paths']['report_json']}`
"""
