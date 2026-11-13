"""DIASTOLE command line interface."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from diastole import __version__
from diastole.config import (
    DEFAULT_OUTPUT_DIR,
    ArrangementConfig,
    AudioConfig,
    EngineConfig,
    MixConfig,
    ReferenceConfig,
    RuntimeConfig,
)
from diastole.interface.dashboard import LiveDashboard
from diastole.runtime.pipeline import Pipeline
from diastole.sensors.simulated import SCENARIOS

console = Console()


def _build_config(
    *,
    model: str,
    scenario: str,
    duration: float | None,
    out: str | None,
    mock: bool,
    audio: bool,
    prefill: bool,
    buffer_seconds: float,
    seed: int,
    temperature: float,
    top_k: int,
    cfg_notes: float,
    cfg_musiccoca: float,
    cfg_drums: float,
    mode: str = "hybrid",
    heartbeat: bool = True,
    heartbeat_pattern: str = "heartbeat",
    heartbeat_gain: float = 0.62,
    accompaniment_gain: float = 0.34,
    mrt2_gain: float = 0.52,
    reference_gain: float = 0.0,
    master_gain: float = 0.92,
    ducking: float = 0.0,
    conditioning: str = "midi",
    style_mode: str = "fixed",
    text_style: str | None = None,
    text_blend: float = 0.65,
    loop_bars: int = 16,
) -> RuntimeConfig:
    engine = EngineConfig(
        model_size=model,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        cfg_drums=cfg_drums,
        prefill=prefill,
    )
    audio_cfg = AudioConfig(enabled=audio, buffer_seconds=buffer_seconds)
    default_text = ReferenceConfig().text_style
    arrangement = ArrangementConfig(
        mode=mode,
        heartbeat=heartbeat,
        heartbeat_pattern=heartbeat_pattern,
        loop_bars=loop_bars,
    )
    mix = MixConfig(
        heartbeat_gain=heartbeat_gain,
        accompaniment_gain=accompaniment_gain,
        mrt2_gain=mrt2_gain,
        reference_gain=reference_gain,
        master_gain=master_gain,
        ducking=ducking,
    )
    reference = ReferenceConfig(
        conditioning_source=conditioning,
        style_mode=style_mode,
        text_style=default_text if text_style is None else text_style,
        text_style_blend=text_blend,
    )
    return RuntimeConfig(
        engine=engine,
        reference=reference,
        arrangement=arrangement,
        mix=mix,
        audio=audio_cfg,
        scenario=scenario,
        duration=duration,
        output_dir=Path(out) if out else DEFAULT_OUTPUT_DIR,
        seed=seed,
    )


def _print_summary(report: dict) -> None:
    if not report:
        console.print("[red]No report was produced.[/red]")
        return
    lat = report["latency_ms"]
    bpm = report["bpm"]
    mus = report["musical"]
    audio = report["audio"]
    pre = report["prefill"]
    table = Table(title="DIASTOLE session summary", show_header=True, header_style="bold cyan")
    table.add_column("Item")
    table.add_column("Result", justify="right")
    table.add_row("Session", report["session"]["name"])
    table.add_row("Mode", report["validation"].get("mode", "legacy"))
    table.add_row("Audio length", f"{report['session']['audio_seconds']:.1f} s")
    table.add_row("BPM range", f"{bpm['min']:.1f} – {bpm['max']:.1f}")
    table.add_row("Tempo range", f"{mus['tempo_min']:.1f} – {mus['tempo_max']:.1f}")
    table.add_row("Mean inference", f"{lat['mean']:.1f} ms/frame (budget 40 ms)")
    table.add_row("p95 inference", f"{lat['p95']:.1f} ms")
    table.add_row("Underruns / dropped", f"{report['buffer']['underruns']} / {report['buffer']['dropped_frames']}")
    table.add_row("Reference windows", str(pre["style_windows"]))
    table.add_row("Prefill", "yes" if pre["used"] else "no")
    table.add_row("Playback", report["playback"]["mode"])
    table.add_row("Audible output", str(report["playback"]["audible"]))
    table.add_row("Note / drum rate", f"{mus['onset_rate_hz']:.2f} / {mus['drum_rate_hz']:.2f} per s")
    table.add_row("Peak / RMS", f"{audio.get('peak', 0):.3f} / {audio.get('rms', 0):.3f}")
    table.add_row("Non-silent WAV", str(report["validation"]["non_silent"]))
    console.print(table)
    console.print(f"[grey62]WAV   {report['paths']['wav']}")
    console.print(f"[grey62]MIDI  {report['paths']['midi']}")
    console.print(f"[grey62]Report {report['paths']['report_json']}")


def _command_reader(pipeline: Pipeline, done: threading.Event) -> None:
    console.print("[grey62]Live commands: bpm <n> | scenario <name> | status | stop[/grey62]")
    while not done.is_set():
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        command = line.strip().lower()
        if not command:
            continue
        parts = command.split()
        try:
            if parts[0] == "bpm" and len(parts) == 2:
                pipeline.set_manual_bpm(float(parts[1]))
                console.print(f"[cyan]manual BPM -> {parts[1]}[/cyan]")
            elif parts[0] == "scenario" and len(parts) == 2:
                pipeline.set_scenario(parts[1])
                console.print(f"[cyan]scenario -> {parts[1]}[/cyan]")
            elif parts[0] == "status":
                console.print(json.dumps(pipeline.snapshot(), indent=2))
            elif parts[0] in ("stop", "quit", "exit"):
                pipeline.stop()
                break
            else:
                console.print("[yellow]Usage: bpm <n> | scenario <name> | status | stop[/yellow]")
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}: {exc}[/red]")


def _execute(
    config: RuntimeConfig,
    *,
    session_name: str | None,
    mock: bool,
    use_reference: bool,
    dashboard: bool,
    interactive: bool,
    initial_bpm: float | None,
) -> dict:
    dash = LiveDashboard(console, enabled=dashboard)
    pipeline = Pipeline(
        config,
        session_name=session_name,
        mock=mock,
        use_reference=use_reference,
        progress=lambda msg: console.print(f"[grey62]{msg}[/grey62]"),
    )
    if initial_bpm is not None:
        pipeline.set_manual_bpm(initial_bpm)

    reader_done = threading.Event()
    reader: threading.Thread | None = None
    if interactive:
        reader = threading.Thread(
            target=_command_reader, args=(pipeline, reader_done), daemon=True
        )
        reader.start()

    try:
        report = pipeline.run(
            duration=config.duration,
            dashboard=dash if dashboard else None,
            dashboard_interval=0.2,
        )
    finally:
        reader_done.set()
        dash.close()
        pipeline.stop()
    _print_summary(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="diastole")
def main() -> None:
    """DIASTOLE — heart-rate-conditioned music with Magenta RealTime 2."""


@main.command()
@click.option("--model", default="mrt2_small", show_default=True)
@click.option("--scenario", type=click.Choice(SCENARIOS), default="rest", show_default=True)
@click.option("--bpm", "initial_bpm", type=float, default=None, help="Manual BPM override.")
@click.option("--duration", type=float, default=None, help="Stop after N seconds.")
@click.option("--out", type=click.Path(file_okay=False), default=None)
@click.option("--mock", is_flag=True, help="Use the model-free oscillator engine.")
@click.option("--no-audio", is_flag=True, help="Do not open an audio device (still records WAV).")
@click.option("--no-prefill", is_flag=True, help="Skip SpectroStream prefill.")
@click.option("--no-reference", is_flag=True, help="Do not use the reference song.")
@click.option("--no-dashboard", is_flag=True, help="Plain text instead of the live dashboard.")
@click.option("--interactive", is_flag=True, help="Read bpm/scenario/stop commands from stdin.")
@click.option("--buffer", "buffer_seconds", type=float, default=1.6, show_default=True)
@click.option("--seed", type=int, default=7, show_default=True)
@click.option("--temperature", type=float, default=1.1, show_default=True)
@click.option("--top-k", type=int, default=50, show_default=True)
@click.option("--cfg-notes", type=float, default=2.0, show_default=True)
@click.option("--cfg-musiccoca", type=float, default=3.0, show_default=True)
@click.option("--cfg-drums", type=float, default=1.0, show_default=True)
@click.option(
    "--mode",
    type=click.Choice(["legacy", "midi", "hybrid"]),
    default="hybrid",
    show_default=True,
    help="legacy=iteration-1 timeline, midi=MIDI accompaniment, hybrid=both+heartbeat.",
)
@click.option("--no-heartbeat", is_flag=True, help="Disable the deterministic heartbeat layer.")
@click.option(
    "--heartbeat-pattern",
    type=click.Choice(["heartbeat", "pulse", "kick"]),
    default="heartbeat",
    show_default=True,
)
@click.option("--heartbeat-gain", type=float, default=0.62, show_default=True)
@click.option("--accompaniment-gain", type=float, default=0.34, show_default=True)
@click.option("--mrt2-gain", type=float, default=0.52, show_default=True)
@click.option("--reference-gain", type=float, default=0.0, show_default=True)
@click.option("--master-gain", type=float, default=0.92, show_default=True)
@click.option("--ducking", type=float, default=0.0, show_default=True)
@click.option(
    "--conditioning",
    type=click.Choice(["midi", "audio"]),
    default="midi",
    show_default=True,
    help="Condition MRT2 on the MIDI render or the original recording.",
)
@click.option(
    "--style-mode",
    type=click.Choice(["fixed", "drift", "loop"]),
    default="fixed",
    show_default=True,
)
@click.option("--text-style", default=None, help="MusicCoCa style text (MIDI conditioning).")
@click.option("--text-blend", type=float, default=0.65, show_default=True)
@click.option("--loop-bars", type=int, default=16, show_default=True)
def run(**kwargs: object) -> None:
    """Run a live heart-rate-conditioned session."""

    config = _build_config(
        model=kwargs["model"],
        scenario=kwargs["scenario"],
        duration=kwargs["duration"],
        out=kwargs["out"],
        mock=kwargs["mock"],
        audio=not kwargs["no_audio"],
        prefill=not kwargs["no_prefill"],
        buffer_seconds=kwargs["buffer_seconds"],
        seed=kwargs["seed"],
        temperature=kwargs["temperature"],
        top_k=kwargs["top_k"],
        cfg_notes=kwargs["cfg_notes"],
        cfg_musiccoca=kwargs["cfg_musiccoca"],
        cfg_drums=kwargs["cfg_drums"],
        mode=kwargs["mode"],
        heartbeat=not kwargs["no_heartbeat"],
        heartbeat_pattern=kwargs["heartbeat_pattern"],
        heartbeat_gain=kwargs["heartbeat_gain"],
        accompaniment_gain=kwargs["accompaniment_gain"],
        mrt2_gain=kwargs["mrt2_gain"],
        reference_gain=kwargs["reference_gain"],
        master_gain=kwargs["master_gain"],
        ducking=kwargs["ducking"],
        conditioning=kwargs["conditioning"],
        style_mode=kwargs["style_mode"],
        text_style=kwargs["text_style"],
        text_blend=kwargs["text_blend"],
        loop_bars=kwargs["loop_bars"],
    )
    _execute(
        config,
        session_name=None,
        mock=bool(kwargs["mock"]),
        use_reference=not kwargs["no_reference"],
        dashboard=not kwargs["no_dashboard"],
        interactive=bool(kwargs["interactive"]),
        initial_bpm=kwargs["initial_bpm"],
    )


@main.command()
@click.option("--duration", type=float, default=42.0, show_default=True)
@click.option("--model", default="mrt2_small", show_default=True)
@click.option("--out", type=click.Path(file_okay=False), default=None)
@click.option("--mock", is_flag=True)
@click.option("--no-audio", is_flag=True)
@click.option("--no-prefill", is_flag=True)
@click.option("--no-dashboard", is_flag=True)
@click.option("--seed", type=int, default=7, show_default=True)
def demo(**kwargs: object) -> None:
    """Scripted acceptance run: rest -> tachycardia -> recovery."""

    config = _build_config(
        model=kwargs["model"],
        scenario="demo",
        duration=kwargs["duration"],
        out=kwargs["out"],
        mock=kwargs["mock"],
        audio=not kwargs["no_audio"],
        prefill=not kwargs["no_prefill"],
        buffer_seconds=1.6,
        seed=kwargs["seed"],
        temperature=1.1,
        top_k=50,
        cfg_notes=2.0,
        cfg_musiccoca=3.0,
        cfg_drums=1.0,
    )
    out_dir = Path(kwargs["out"]) if kwargs["out"] else DEFAULT_OUTPUT_DIR / "demos"
    config = dataclasses.replace(config, output_dir=out_dir)
    session_name = time.strftime("demo_%Y%m%d_%H%M%S")
    _execute(
        config,
        session_name=session_name,
        mock=bool(kwargs["mock"]),
        use_reference=True,
        dashboard=not kwargs["no_dashboard"],
        interactive=False,
        initial_bpm=None,
    )


# Controlled comparison used for the iteration-2 evaluation.  The ladder moves
# one variable at a time: iteration-1 baseline -> deterministic MIDI material ->
# heartbeat -> MIDI-rendered conditioning -> the full hybrid.
_EXPERIMENTS: dict[str, dict] = {
    "A": {
        "label": "baseline",
        "note": "Iteration-1 baseline: MRT2 from the audio reference, no heartbeat or MIDI material.",
        "mode": "legacy",
        "heartbeat": False,
        "conditioning": "audio",
        "style_mode": "drift",
        "text_blend": 0.0,
        "heartbeat_gain": 0.0,
        "accompaniment_gain": 0.0,
        "mrt2_gain": 0.52,
        "reference": True,
    },
    "B": {
        "label": "midi_material",
        "note": "MIDI analysis only: deterministic instrumental arrangement, no MRT2 or heartbeat.",
        "mode": "midi",
        "heartbeat": False,
        "conditioning": "audio",
        "style_mode": "fixed",
        "text_blend": 0.0,
        "heartbeat_gain": 0.0,
        "accompaniment_gain": 0.95,
        "mrt2_gain": 0.0,
        "reference": False,
    },
    "C": {
        "label": "heartbeat_mrt2",
        "note": "Heartbeat + the iteration-1 MRT2 timeline, audio conditioning.",
        "mode": "legacy",
        "heartbeat": True,
        "conditioning": "audio",
        "style_mode": "drift",
        "text_blend": 0.0,
        "heartbeat_gain": 0.62,
        "accompaniment_gain": 0.0,
        "mrt2_gain": 0.52,
        "reference": True,
    },
    "D": {
        "label": "midi_conditioning",
        "note": "Iteration-1 MRT2 timeline conditioned on the instrumental MIDI render + style text.",
        "mode": "legacy",
        "heartbeat": False,
        "conditioning": "midi",
        "style_mode": "fixed",
        "text_blend": 0.55,
        "heartbeat_gain": 0.0,
        "accompaniment_gain": 0.0,
        "mrt2_gain": 0.52,
        "reference": True,
    },
    "E": {
        "label": "full_hybrid",
        "note": "Full hybrid: heartbeat + MIDI accompaniment + MRT2 conditioned on the MIDI render.",
        "mode": "hybrid",
        "heartbeat": True,
        "conditioning": "midi",
        "style_mode": "fixed",
        "text_blend": 0.65,
        "heartbeat_gain": 0.62,
        "accompaniment_gain": 0.34,
        "mrt2_gain": 0.50,
        "reference": True,
    },
}


@main.command()
@click.option("--duration", type=float, default=15.0, show_default=True)
@click.option("--out", type=click.Path(file_okay=False), default=None)
@click.option("--model", default="mrt2_small", show_default=True)
@click.option("--seed", type=int, default=7, show_default=True)
@click.option("--bpm", type=float, default=70.0, show_default=True, help="Fixed BPM for controlled comparison.")
@click.option("--no-audio", is_flag=True, help="Record WAVs without opening a device.")
@click.option("--mock", is_flag=True, help="Use the oscillator engine (runner check only).")
@click.option("--only", default=None, help="Comma-separated experiment keys, e.g. A,C,E.")
def experiments(
    duration: float,
    out: str | None,
    model: str,
    seed: int,
    bpm: float,
    no_audio: bool,
    mock: bool,
    only: str | None,
) -> None:
    """Run controlled experiments A-E and write audio/report/metrics per run."""

    keys = (
        [k.strip().upper() for k in only.split(",") if k.strip()]
        if only
        else list(_EXPERIMENTS)
    )
    unknown = [k for k in keys if k not in _EXPERIMENTS]
    if unknown:
        raise click.BadParameter(f"unknown experiment(s): {', '.join(unknown)}")

    root = Path(out) if out else DEFAULT_OUTPUT_DIR / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for key in keys:
        spec = _EXPERIMENTS[key]
        console.rule(f"[bold]Experiment {key} — {spec['label']}[/bold]")
        console.print(f"[grey62]{spec['note']}[/grey62]")
        config = _build_config(
            model=model,
            scenario="manual",
            duration=duration,
            out=str(root),
            mock=False,
            audio=not no_audio,
            prefill=True,
            buffer_seconds=1.6,
            seed=seed,
            temperature=1.1,
            top_k=50,
            cfg_notes=2.0,
            cfg_musiccoca=3.0,
            cfg_drums=1.0,
            mode=spec["mode"],
            heartbeat=spec["heartbeat"],
            heartbeat_gain=spec["heartbeat_gain"],
            accompaniment_gain=spec["accompaniment_gain"],
            mrt2_gain=spec["mrt2_gain"],
            ducking=0.0,
            conditioning=spec["conditioning"],
            style_mode=spec["style_mode"],
            text_blend=spec["text_blend"],
            loop_bars=16,
        )
        try:
            report = _execute(
                config,
                session_name=f"exp_{key}_{spec['label']}",
                mock=mock,
                use_reference=spec["reference"],
                dashboard=False,
                interactive=False,
                initial_bpm=bpm,
            )
        except Exception as exc:
            console.print(f"[red]Experiment {key} failed: {type(exc).__name__}: {exc}[/red]")
            rows.append({"key": key, "label": spec["label"], "error": str(exc)})
            continue

        row = _experiment_metrics(key, spec, report)
        row["wav"] = report["paths"]["wav"]
        row["report"] = report["paths"]["report_json"]
        rows.append(row)

        # Keep the instrumental reference next to the mix for auditioning.
        rendered = (report.get("reference") or {}).get("rendered_audio")
        if rendered and Path(rendered).exists():
            import shutil

            stem = root / f"exp_{key}_{spec['label']}_midi_render.wav"
            shutil.copyfile(rendered, stem)
            row["midi_render"] = str(stem)

    _write_experiment_summary(root, rows, duration, seed, bpm)
    _print_experiment_table(rows)
    console.print(f"[grey62]Artifacts: {root}[/grey62]")


def _experiment_metrics(key: str, spec: dict, report: dict) -> dict:
    audio = report.get("audio", {})
    buffer = report.get("buffer", {})
    latency = report.get("latency_ms", {})
    musical = report.get("musical", {})
    arrangement = report.get("arrangement", {})
    layers = arrangement.get("layers") or {}
    return {
        "key": key,
        "label": spec["label"],
        "mode": spec["mode"],
        "conditioning": spec["conditioning"],
        "heartbeat": spec["heartbeat"],
        "bpm": round(float(report.get("bpm", {}).get("mean", 0.0)), 2),
        "audio_seconds": round(float(report.get("session", {}).get("audio_seconds", 0.0)), 2),
        "peak": round(float(audio.get("peak", 0.0)), 4),
        "rms": round(float(audio.get("rms", 0.0)), 4),
        "clipped_samples": int(audio.get("clipped_samples", 0)),
        "onset_rate_hz": round(float(musical.get("onset_rate_hz", 0.0)), 3),
        "underruns": int(buffer.get("underruns", 0)),
        "dropped_frames": int(buffer.get("dropped_frames", 0)),
        "mean_latency_ms": round(float(latency.get("mean", 0.0)), 2),
        "p95_latency_ms": round(float(latency.get("p95", 0.0)), 2),
        "limiter_reductions": int(layers.get("limiter_reductions", 0)),
        "heartbeat_beats": int(layers.get("heartbeat_beat_events", 0)),
        "analysis_key": arrangement.get("key", "n/a"),
    }


def _write_experiment_summary(
    root: Path, rows: list[dict], duration: float, seed: int, bpm: float
) -> None:
    summary = {
        "duration_seconds": duration,
        "seed": seed,
        "fixed_bpm": bpm,
        "experiments": rows,
    }
    path = root / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Experiments A–E",
        "",
        f"Duration {duration:.1f} s per run, seed {seed}, fixed manual BPM {bpm:.0f}.",
        "",
        "| Exp | Configuration | Mode | BPM | Peak | RMS | Onsets/s | Clips | Underruns |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['key']} | {row['label']} | failed | — | — | — | — | — | — |")
            continue
        lines.append(
            "| {key} | {label} | {mode} | {bpm:.1f} | {peak:.3f} | {rms:.3f} | "
            "{onset_rate_hz:.2f} | {clipped_samples} | {underruns} |".format(**row)
        )
    lines += [
        "",
        "## Listening notes",
        "",
        "_Fill in after auditioning each WAV (the agent cannot evaluate audio subjectively)._",
        "",
    ]
    (root / "EXPERIMENTS.md").write_text("\n".join(lines), encoding="utf-8")


def _print_experiment_table(rows: list[dict]) -> None:
    table = Table(title="Experiments A–E", header_style="bold cyan")
    table.add_column("Exp")
    table.add_column("Configuration")
    table.add_column("Mode")
    table.add_column("BPM", justify="right")
    table.add_column("Peak", justify="right")
    table.add_column("RMS", justify="right")
    table.add_column("Onsets/s", justify="right")
    table.add_column("Clips", justify="right")
    table.add_column("Underruns", justify="right")
    for row in rows:
        if "error" in row:
            table.add_row(row["key"], row["label"], "failed", "—", "—", "—", "—", "—", "—")
            continue
        table.add_row(
            row["key"],
            row["label"],
            row["mode"],
            f"{row['bpm']:.1f}",
            f"{row['peak']:.3f}",
            f"{row['rms']:.3f}",
            f"{row['onset_rate_hz']:.2f}",
            str(row["clipped_samples"]),
            str(row["underruns"]),
        )
    console.print(table)


@main.command()
@click.option("--force", is_flag=True, help="Rebuild cached tokens/state.")
@click.option("--no-state", is_flag=True, help="Only prepare style tokens and pitch profile.")
@click.option("--model", default="mrt2_small", show_default=True)
def prepare(force: bool, no_state: bool, model: str) -> None:
    """Precompute MusicCoCa style windows and SpectroStream prefill state."""

    from diastole.config import ReferenceConfig
    from diastole.generation.engine import MRT2Engine
    from diastole.music.reference import ReferenceMusic

    reference = ReferenceMusic(ReferenceConfig())
    console.print("[bold]Loading reference audio...[/bold]")
    reference.load_audio()
    console.print(f"Duration: {reference.duration:.2f} s at {reference.sample_rate} Hz")
    tokens = reference.prepare_style_tokens(force=force, progress=lambda m: console.print(m))
    console.print(f"Style windows: {tokens.shape} -> {reference.style_cache_path}")
    profile = reference.pitch_profile(force=force)
    console.print(f"Pitch profile: {profile.as_dict()['top_pitch_classes']} from {profile.source}")
    if not no_state:
        console.print("[bold]Loading MRT2 and prefilling...[/bold]")
        engine = MRT2Engine(EngineConfig(model_size=model), warmup=False)
        state = reference.prefill_state(engine, force=force, progress=lambda m: console.print(m))
        console.print(f"Prefill state: {len(state)} arrays")
    console.print("[green]Preparation complete.[/green]")


@main.command("inspect-reference")
@click.option("--prepare", is_flag=True, help="Build the MusicCoCa cache if missing.")
def inspect_reference(prepare: bool) -> None:
    """Show what the reference song contributes to generation."""

    from diastole.config import ReferenceConfig
    from diastole.music.reference import ReferenceMusic

    reference = ReferenceMusic(ReferenceConfig())
    reference.load_audio()
    console.print_json(json.dumps(reference.info()))
    table = Table(title="MusicCoCa style windows", header_style="bold cyan")
    table.add_column("session time")
    table.add_column("song window")
    table.add_column("12 RVQ style tokens")
    try:
        if prepare or reference.style_cache_path.exists():
            reference.prepare_style_tokens()
            for t in (0.0, 10.0, 20.0, 30.0):
                win = reference.style_window_at(t)
                table.add_row(
                    f"{t:.1f}s",
                    f"{win['start_seconds']:.1f}–{win['end_seconds']:.1f}s",
                    ", ".join(str(v) for v in win["tokens"]),
                )
            console.print(table)
        else:
            console.print("[yellow]No style cache yet; run `diastole prepare`.[/yellow]")
    except Exception as exc:
        console.print(f"[red]Style cache unavailable: {exc}[/red]")


@main.command()
@click.option("--full", is_flag=True, help="Also load MRT2 and generate two frames.")
@click.option("--model", default="mrt2_small", show_default=True)
def selftest(full: bool, model: str) -> None:
    """Check assets, MLX import, reference material and audio devices."""

    from diastole import _magenta  # noqa: F401  (sets MAGENTA_HOME before paths)
    from diastole.config import ensure_magenta_home

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    assets = ensure_magenta_home()
    check("MAGENTA_HOME", assets.exists(), str(assets))
    from diastole._magenta import mx, paths

    model_path = paths.models_dir() / model / f"{model}.mlxfn"
    state_path = paths.models_dir() / model / f"{model}_state.safetensors"
    check("model .mlxfn", model_path.exists(), str(model_path))
    check("model state", state_path.exists(), str(state_path))
    encoder = paths.spectrostream_dir() / "spectrostream_encoder.mlxfn"
    check("SpectroStream encoder", encoder.exists(), str(encoder))
    musiccoca_files = list(paths.musiccoca_dir().glob("*.tflite"))
    check("MusicCoCa resources", len(musiccoca_files) >= 4, f"{len(musiccoca_files)} tflite files")

    from diastole.config import REFERENCE_AUDIO, REFERENCE_MIDI

    check("reference audio", REFERENCE_AUDIO.exists(), str(REFERENCE_AUDIO))
    check("reference MIDI", REFERENCE_MIDI.exists(), str(REFERENCE_MIDI))

    try:
        fn = mx.import_function(str(model_path))
        check("MLX import_function", True, f"mlx {mx.__version__ if hasattr(mx, '__version__') else '?'}")
    except Exception as exc:
        fn = None
        check("MLX import_function", False, f"{type(exc).__name__}: {exc}")

    try:
        import sounddevice as sd

        devices = sd.query_devices()
        outputs = [d for d in devices if d.get("max_output_channels", 0) > 0]
        check("audio output devices", bool(outputs), f"{len(outputs)} device(s)")
    except Exception as exc:
        check("audio output devices", False, f"{type(exc).__name__}: {exc}")

    try:
        import numpy as np
        import soundfile as sf

        test_path = DEFAULT_OUTPUT_DIR / "selftest.wav"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(4800, dtype=np.float32) / 48000.0
        sf.write(test_path, 0.1 * np.sin(2 * np.pi * 440 * t), 48000)
        check("WAV write", test_path.exists(), str(test_path))
    except Exception as exc:
        check("WAV write", False, f"{type(exc).__name__}: {exc}")

    if full:
        try:
            from diastole.generation.engine import MRT2Engine

            engine = MRT2Engine(EngineConfig(model_size=model), warmup=True)
            style = [-1] * 12
            notes = [0] * 128
            notes[60] = 2
            peaks = []
            for _ in range(3):
                samples, ms = engine.step(style, notes, (0,))
                peaks.append(float(abs(samples).max()))
            peak = max(peaks)
            check("MRT2 streaming step", peak > 1e-4, f"peak {peak:.4f}, {ms:.1f} ms/frame")
        except Exception as exc:
            check("MRT2 streaming step", False, f"{type(exc).__name__}: {exc}")

    table = Table(title="DIASTOLE selftest", header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail", overflow="fold")
    for name, ok, detail in results:
        table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(table)
    failed = [r for r in results if not r[1]]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
