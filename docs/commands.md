# Commands

All commands run from the repository root with `uv`.

## Setup

```bash
./scripts/link_assets.sh        # symlink existing Magenta models/resources into assets/
uv sync --extra dev             # create .venv, install pinned mlx==0.31.1 etc.
uv run diastole --help
```

`scripts/link_assets.sh` never downloads or copies model files. Set
`MAGENTA_HOME` only if you want to use a different asset root; the default is the
repository's `assets/` directory. The code appends `magenta-rt-v2` itself.

## `diastole selftest`

Fast structural checks: `MAGENTA_HOME`, model/state files, SpectroStream
encoder, MusicCoCa resources, reference audio/MIDI, MLX import, output devices
and WAV writing.

```bash
uv run diastole selftest
uv run diastole selftest --full   # also loads MRT2 and generates 3 frames
```

`--full` is the quickest real-model check; it verifies a non-silent peak and
prints ms/frame.

## `diastole prepare`

Builds the reference caches (not required if `.cache/` is present):

```bash
uv run diastole prepare                 # style windows + pitch profile + prefill state
uv run diastole prepare --no-state      # skip SpectroStream/prefill
uv run diastole prepare --force         # rebuild everything
uv run diastole prepare --model mrt2_small
```

First build costs roughly 30 s for SpectroStream encoding plus 26 s for the
650-frame prefill. The resulting 165-array state is cached in `.cache/` and loads
in ~0.01 s on later runs. The JSON sidecar records the original compute time.

## `diastole inspect-reference`

Shows what the reference song contributes: audio metadata, pitch profile and
the 12-token MusicCoCa windows at selected session times.

```bash
uv run diastole inspect-reference
uv run diastole inspect-reference --prepare
```

## `diastole demo`

Scripted acceptance run: rest → tachycardia → recovery. Default: 42 s, real
model, prefill, reference conditioning, live dashboard, audio output.

```bash
uv run diastole demo --duration 42
uv run diastole demo --duration 42 --no-dashboard --out outputs/final_demo
uv run diastole demo --mock --no-audio --no-dashboard   # fast, no model
```

| Option | Meaning |
|---|---|
| `--duration N` | stop after N seconds (default 42) |
| `--model NAME` | model directory under `assets/magenta-rt-v2/models` |
| `--out DIR` | output root; a timestamped session directory is created |
| `--mock` | use the model-free oscillator engine |
| `--no-audio` | do not open a device (still records WAV and measures buffers) |
| `--no-prefill` | skip SpectroStream state injection |
| `--no-dashboard` | plain progress lines instead of the live dashboard |
| `--seed N` | scheduler/engine seed (default 7) |

`demo --mock` still exercises the real signal chain, scheduler, adapter, buffers,
reporting and WAV/MIDI writing.

## `diastole run`

Interactive or fixed-duration session with the same engine and audio path.

```bash
uv run diastole run
uv run diastole run --scenario workout --duration 60
uv run diastole run --bpm 120 --no-dashboard
uv run diastole run --no-audio --duration 10 --out outputs/record_only
```

| Option | Meaning |
|---|---|
| `--scenario NAME` | `rest`, `gradual`, `sudden`, `workout`, `demo`, `random`, `manual` |
| `--bpm N` | manual BPM override at start |
| `--duration N` | stop after N seconds (omit for open-ended) |
| `--mock` | model-free engine |
| `--no-audio` | record-only simulated sink |
| `--no-prefill` | cold start with style conditioning only |
| `--no-reference` | no MusicCoCa/prefill; masked style tokens |
| `--no-dashboard` | plain text |
| `--interactive` | read commands from stdin |
| `--buffer S` | ring buffer seconds (default 1.6) |
| `--seed N` | deterministic seed |
| `--mode legacy\|midi\|hybrid` | iteration-1 scheduler / MIDI accompaniment / MIDI + heartbeat (default hybrid) |
| `--no-heartbeat` | disable the deterministic heartbeat layer |
| `--heartbeat-pattern heartbeat\|pulse\|kick` | heartbeat figure |
| `--heartbeat-gain`, `--accompaniment-gain`, `--mrt2-gain`, `--reference-gain`, `--master-gain` | per-layer mix gains |
| `--ducking 0..1` | sidechain MRT2/accompaniment from the heartbeat |
| `--conditioning midi\|audio` | condition MRT2 on the MIDI render (default) or MP3 |
| `--style-mode fixed\|drift\|loop` | how the MusicCoCa window advances |
| `--text-style TEXT`, `--text-blend 0..1` | MusicCoCa text style and its blend weight |
| `--loop-bars N` | MIDI loop length (default 16) |
| `--temperature`, `--top-k`, `--cfg-notes`, `--cfg-musiccoca`, `--cfg-drums` | MRT2 sampling/CFG controls |

Interactive commands:

```
bpm 120
scenario sudden
status
stop
```

## `diastole experiments`

Runs the controlled A–E ladder with a fixed manual BPM so the comparison is not
confounded by the physiological scenario:

| Key | Configuration |
|---|---|
| A | iteration-1 baseline (MRT2 from the audio reference, no heartbeat/MIDI material) |
| B | MIDI analysis only (deterministic instrumental render, MRT2 silent) |
| C | heartbeat + iteration-1 MRT2 timeline, audio conditioning |
| D | iteration-1 MRT2 timeline conditioned on the MIDI render + style text |
| E | full hybrid: heartbeat + MIDI accompaniment + MIDI-conditioned MRT2 |

```bash
uv run diastole experiments --duration 15 --bpm 70 --out outputs/experiments
uv run diastole experiments --only A,C,E
uv run diastole experiments --mock --no-audio --duration 2   # runner check
```

Each run writes `audio.wav`, `score.mid`, `report.json`, `midi_analysis.json`
(where applicable), and the suite writes `summary.json` and an
`EXPERIMENTS.md` table. Where a run uses the MIDI reference, its rendered
instrumental stem is copied next to the mix for auditioning.

## Outputs

Each run creates `outputs/<session-name>/` (or under `--out`):

| File | Description |
|---|---|
| `audio.wav` | 48 kHz, 16-bit stereo recording of the mixed output. |
| `score.mid` | One type-1 SMF for the whole session (tempo map, notes, drums). |
| `midi_analysis.json` | Key, chords, bars, selected loop, theme and diagnostics. |
| `report.json` | Full measurements and per-frame time series. |
| `report.md` | Human-readable summary. |

### Report fields

- `session` — name, wall/audio seconds, frames, mock flag.
- `engine` — model, frame geometry, state size, load/warm-up time, CFG scales.
- `arrangement` — mode, key/confidence, BPM, loop, chords, theme notes,
  diagnostics, render summary, arranger state and mixer layer stats.
- `prefill` — used, original compute seconds, cache-load seconds, cache hit/path,
  style-window count, error.
- `latency_ms` — mean/p50/p95/max, 40 ms budget, real-time factor.
- `playback` — backend mode, active/audible flags, frames/blocks played,
  underruns, device error.
- `buffer` — capacity, max/mean fill, underruns, pre-roll frames, dropped frames.
- `bpm` — raw and smoothed min/max/mean/median.
- `musical` — tempo/density/activity/dynamics, active notes, onset and drum
  rates overall and for high/low BPM segments, MIDI event rates.
- `timeline` — note/drum/tempo event counts, duration, pitch-class histogram.
- `generation_gaps_s` — steady-state gap mean/p95/max, plus the startup/handoff
  gap (excluded from steady-state stats).
- `audio` — sample rate, frames, peak, RMS, clipped samples, NaN/silence flags.
- `validation` — non-silent, continuous frames, MIDI written, prefilled, audible.
- `timeseries` — one record per generated frame (bounded to the session).
