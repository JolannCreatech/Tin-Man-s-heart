"""Live terminal dashboard (rich)."""

from __future__ import annotations

import sys

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _bar(value: float, width: int = 18, color: str = "cyan") -> Text:
    value = max(0.0, min(1.0, float(value)))
    filled = round(value * width)
    text = Text()
    text.append("█" * filled, style=color)
    text.append("░" * (width - filled), style="grey37")
    return text


class LiveDashboard:
    """Callable dashboard: ``dashboard(snapshot)`` updates the Live display."""

    def __init__(self, console: Console | None = None, enabled: bool = True) -> None:
        self.console = console or Console()
        self.enabled = enabled and sys.stdout.isatty()
        self._live: Live | None = None

    def __call__(self, snapshot: dict) -> None:
        if not self.enabled:
            return
        if self._live is None:
            self._live = Live(
                self.render(snapshot),
                console=self.console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
        else:
            self._live.update(self.render(snapshot))

    def render(self, snap: dict) -> Panel:
        latest = snap.get("latest", {})
        bpm_raw = latest.get("bpm_raw", 0.0)
        bpm = latest.get("bpm", 0.0)
        arousal = latest.get("arousal", 0.0)
        tempo = latest.get("tempo", 0.0)
        density = latest.get("density", 0.0)
        activity = latest.get("note_activity", 0.0)
        intensity = latest.get("intensity", 0.0)
        dynamics = latest.get("dynamics", 0.0)
        active = latest.get("active_notes", 0)
        onsets = latest.get("onsets", 0)
        subdivision = latest.get("subdivision", 0)

        color = "green" if arousal < 0.4 else ("yellow" if arousal < 0.7 else "red")

        heart = Table.grid(expand=True)
        heart.add_column(ratio=1)
        heart.add_column(ratio=1)
        heart.add_column(ratio=1)
        heart.add_column(ratio=1)
        heart.add_row(
            Text.assemble(("HEART\n", "bold grey62"), (f"{bpm_raw:5.1f}", "bold white"), (" raw bpm", "grey62")),
            Text.assemble(("SMOOTH\n", "bold grey62"), (f"{bpm:5.1f}", f"bold {color}"), (" bpm", "grey62")),
            Text.assemble(("TEMPO\n", "bold grey62"), (f"{tempo:5.1f}", "bold cyan"), (" bpm", "grey62")),
            Text.assemble(("AROUSAL\n", "bold grey62"), _bar(arousal, color=color), (f" {arousal:0.2f}", "grey62")),
        )

        music = Table.grid(expand=True)
        music.add_column(ratio=1)
        music.add_column(ratio=1)
        music.add_row(
            Text.assemble(("density     ", "grey62"), _bar(density), (f" {density:0.2f}", "grey70")),
            Text.assemble(("note act.   ", "grey62"), _bar(activity), (f" {activity:0.2f}", "grey70")),
        )
        music.add_row(
            Text.assemble(("drums       ", "grey62"), _bar(intensity), (f" {intensity:0.2f}", "grey70")),
            Text.assemble(("dynamics    ", "grey62"), _bar(dynamics), (f" {dynamics:0.2f}", "grey70")),
        )

        perf = Table.grid(expand=True)
        perf.add_column(ratio=1)
        perf.add_column(ratio=1)
        perf.add_row(
            Text.assemble(("inference\n", "bold grey62"), (f"{snap.get('latency_mean_ms', 0):5.1f} ms", "bold magenta"), (" mean", "grey62")),
            Text.assemble(("p95 / budget\n", "bold grey62"), (f"{snap.get('latency_p95_ms', 0):5.1f}", "bold magenta"), (" / 40 ms", "grey62")),
        )
        perf.add_row(
            Text.assemble(("buffer\n", "bold grey62"), _bar(snap.get("buffer_fraction", 0.0), color="blue"), (f" {snap.get('buffer_seconds', 0):0.2f}s", "grey62")),
            Text.assemble(("underruns\n", "bold grey62"), (f"{snap.get('underruns', 0)}", "bold yellow" if snap.get("underruns", 0) else "bold green"), ("  dropped ", "grey62"), (f"{snap.get('dropped_frames', 0)}", "grey70")),
        )

        state = Table.grid(expand=True)
        state.add_column(ratio=1)
        state.add_column(ratio=1)
        state.add_row(
            Text.assemble(("notes ", "grey62"), (f"{snap.get('notes_total', 0)}", "white"), ("  drums ", "grey62"), (f"{snap.get('drums_total', 0)}", "white"), ("  active ", "grey62"), (f"{active}", "white"), ("  onsets ", "grey62"), (f"{onsets}", "white")),
            Text.assemble(("grid 1/", "grey62"), (f"{subdivision}", "white"), ("  frame ", "grey62"), (f"{snap.get('frames', 0)}", "white"), ("  t=", "grey62"), (f"{latest.get('t', 0):0.1f}s", "white")),
        )

        footer = Text()
        footer.append(f"sensor {snap.get('sensor_status', '?')}  ", style="grey62")
        footer.append(f"audio {snap.get('audio_mode', '?')}  ", style="grey62")
        footer.append(f"prefill {'on' if snap.get('prefill') else 'off'}  ", style="grey62")
        if snap.get("reference_error"):
            footer.append(f"\n{snap['reference_error']}", style="bold red")
        footer.append("\ncommands: bpm <n> · scenario <name> · status · stop", style="grey50")

        title = Text.assemble(("DIASTOLE ", "bold white"), ("— If I Only Had a Heart", "italic grey62"))
        return Panel(
            Group(heart, Text(""), music, Text(""), perf, Text(""), state, Text(""), Align.left(footer)),
            title=title,
            border_style=color,
            padding=(0, 1),
        )

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
