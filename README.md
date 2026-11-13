# DIASTOLE

Real-time music generation driven by a heartbeat and by the song
*“If I Only Had a Heart”* (The Wizard of Oz), using **Magenta RealTime 2** on
Apple Silicon.

> Status: **working prototype, verified end-to-end.** Iteration 1 generated and
> played continuous audio at 25.4 ms/frame mean (40 ms budget), with 0
> underruns, 0 dropped frames and a full-session WAV/MIDI/report. **Iteration 2**
> adds a deterministic BPM-locked heartbeat, a MIDI-derived instrumental
> arrangement and a multi-layer mixer, and conditions MRT2 on an instrumental
> rendering of the transcription instead of the vocal MP3. See
> `IMPLEMENTATION_LOG.md`, `MIDI_ANALYSIS.md`, `AUDIO_INVESTIGATION.md` and
> `EXPERIMENTS.md` for measurements; `docs/` for architecture and limits.

## What it does

1. A physiological sensor connector produces heart-rate measurements. The
   included source is a realistic simulated sensor (OU noise, respiratory sinus
   arrhythmia, scripted scenarios, manual overrides); the sensor interface is
   ready for a real BLE/serial device.
2. The BPM signal is validated, filtered and mapped to musical parameters
   (tempo, rhythmic density, energy, dynamics, rhythmic intensity).
3. Those parameters drive one continuously evolving MIDI/control timeline. The
   same event objects are converted to the 25 Hz note-onset and drum tensors
   that MRT2 actually consumes. MIDI is the inspectable score, not the model
   input, and is never regenerated per sensor sample.
4. The song conditions generation in two genuine ways:
   - **MusicCoCa style tokens** from 10 s windows, now derived by default from
     an *instrumental rendering of the MIDI transcription* (optionally blended
     with a text style prompt), so the vocal MP3 no longer drives the timbre.
   - **SpectroStream prefill**: a 28 s excerpt of that render is encoded to RVQ
     tokens and fed through the exported transformer, seeding its streaming
     state. Generation continues that musical context rather than starting cold.
5. MRT2 generates 40 ms stereo frames continuously in one persistent streaming
   state; a ring-buffered playback thread sends them to the system output and a
   recorder writes the session WAV.

The raw recording is never mixed into the output.

## Iteration 2: three separable layers

The default `hybrid` mode mixes three layers with independent gains:

| Layer | Source | Runs in | Role |
|---|---|---|---|
| **A. Heartbeat** | `HeartbeatSynth` from the live BPM | playback thread | exact lub-dub pulse that survives inference stalls |
| **B. MRT2** | `mrt2_small`, MIDI-render conditioning | generation thread | harmonic/textural colour |
| **C. MIDI arrangement** | analysed 16-bar loop → `MidiArranger` | generation schedules, playback renders | bass, chord pad, lead theme |

MRT2 receives only bass and pad notes (the lead melody and drum pulse are kept
away from it). The analysed transcription is Eb major; the automatic loop is
bars 44–59 (Eb–Ab–Bb–Eb, 32 s). The mixer applies smoothed per-layer gains, an
optional heartbeat sidechain duck, a limiter and a clip guard, and records the
actual mixed output. The raw MP3 is never mixed in.

## Quick start (macOS Apple Silicon, Python 3.12)

```bash
# 1. Link the already-downloaded Magenta assets (symlinks, no downloads).
./scripts/link_assets.sh

# 2. Install the environment with uv.
uv sync --extra dev

# 3. Fast checks + real model smoke test.
uv run diastole selftest
uv run diastole selftest --full

# 4. Run the scripted acceptance demo (rest -> tachycardia -> recovery).
uv run diastole demo --duration 42

# 5. Or run interactively.
uv run diastole run
```

While running interactively:

```
bpm 120          # change simulated heart rate
scenario sudden  # rest | gradual | sudden | workout | demo | random | manual
status           # print a snapshot
stop             # graceful shutdown
```

Useful variants:

```bash
uv run diastole demo --mock --no-audio --no-dashboard   # no model, fast pipeline test
uv run diastole run --no-audio                          # record only, no device
uv run diastole run --mode hybrid --conditioning midi --duration 42
uv run diastole run --mode midi --mrt2-gain 0 --no-heartbeat   # MIDI material only
uv run diastole run --mode legacy --conditioning audio          # iteration-1 ablation
uv run diastole run --heartbeat-gain 0.8 --accompaniment-gain 0.3 --mrt2-gain 0.4
uv run diastole run --no-prefill                        # style tokens, cold state
uv run diastole experiments --duration 15 --out outputs/experiments  # A-E suite
uv run diastole prepare                                 # rebuild reference caches
uv run diastole inspect-reference                       # show style windows/tokens
```

`--mode` is `legacy` (iteration-1 scheduler), `midi` (deterministic MIDI
accompaniment, no heartbeat required) or `hybrid` (MIDI + heartbeat + MRT2).
`--conditioning` selects `midi` (instrumental render, default) or `audio` (the
original MP3, for comparison). Per-layer gains are `--heartbeat-gain`,
`--accompaniment-gain`, `--mrt2-gain`, `--reference-gain` and `--master-gain`.

Outputs go to `outputs/<session>/`: `audio.wav`, `score.mid`, `report.json`,
`report.md`. The report contains measured latency, buffer/underrun statistics,
BPM and musical response, note/drum event rates, audio validation and the full
per-frame time series.

## Measured acceptance result

Final 42 s run with the real `mrt2_small` model, cached prefill and CoreAudio
output (`outputs/final_demo/`):

| Metric | Result |
|---|---|
| Audio length | 42.0 s (1051 frames) |
| Mean / p95 / max inference | 25.4 / 28.6 / 53.9 ms per 40 ms frame |
| Real-time factor | 0.64 |
| Underruns / dropped frames | 0 / 0 |
| Playback | `sounddevice` (audible) |
| Peak / RMS | 0.876 / 0.116 (0 clipped samples) |
| BPM range (smoothed) | 62.2 – 133.3 |
| Note onset rate, high vs low BPM | 3.43 vs 0.63 per s |
| Drum-active frame rate, high vs low BPM | 2.23 vs 0.05 per s |
| Prefill | cached, 26.4 s original compute, 0.013 s load |

`uv run pytest -q`: **54 passed** (includes the iteration-2 layer tests).

Iteration-2 real-MRT2 hybrid check (12 s, MIDI conditioning, no audio device):

| Metric | Result |
|---|---|
| Conditioning | MIDI render, 10 MusicCoCa windows, text blend 0.65 |
| Prefill | 650-frame MIDI-render excerpt, cached |
| Inference | 25.5 ms mean / 27.5 ms p95 |
| Underruns / dropped | 0 / 0 |
| Audio | 12.04 s, peak 0.654, RMS 0.077, 0 clipped |
| Low-band events | 13 at ~62 BPM (matches the beat count) |

Final real-device hybrid run (20 s, fixed 70 BPM, CoreAudio):

| Metric | Result |
|---|---|
| Playback | `sounddevice`, audible |
| Audio | 20.04 s, peak 0.628, RMS 0.076, 0 clipped |
| Heartbeat | 26 events, low-band period 0.7951 s (expected 0.7947 s) |
| Inference | 25.8 ms mean / 27.2 ms p95, 0 underruns |

A second real-model session verified the live control path: sending manual BPM
145 then 75 during playback moved smoothed BPM 62 → 140.6 → 84, tempo
70 → 121 → 85, with 3.35 vs 0.85 onsets/s and 2.23 vs 0.28 drum-active
frames/s in high vs low segments, and 0 underruns. See
`IMPLEMENTATION_LOG.md`. Controlled A–E comparisons are in `EXPERIMENTS.md`.

## Requirements and assets

- Python 3.12, `uv`, macOS Apple Silicon.
- `mlx==0.31.1` (pinned). MLX 0.32.x cannot import the shipped `.mlxfn`.
- The existing local Magenta RealTime 2 assets (model + MusicCoCa +
  SpectroStream) and the reference MP3/MIDI. `scripts/link_assets.sh` creates
  `assets/magenta-rt-v2/{resources,models}` symlinks; nothing external is
  modified or duplicated.
- `sounddevice` uses CoreAudio directly; no Homebrew PortAudio package is
  required on this machine. Without an output device the same pipeline runs a
  paced simulated sink and still records the WAV.

The C++ standalone Magenta runtime is **not** used: the active command-line
tools have no Metal compiler (`xcrun: unable to find utility "metal"`). The
Python/MLX path is the supported path here.

## Repository layout

```
src/diastole/
  audio/        ring buffer, playback thread, WAV recorder, synths, mixer
  generation/   MRT2 engine wrapper, prefill, mock engine
  interface/    CLI, experiments runner and Rich dashboard
  music/        timeline, MIDI writer, 25 Hz adapter, MIDI analysis/render,
                arranger, reference caches
  runtime/      pipeline orchestration and reporting
  sensors/      sensor ABC, simulated heart-rate source
  signal/       BPM processing and physiological -> musical mapping
docs/           architecture, mapping, commands, testing, limitations
tests/          54 unit/integration tests (mock engine + CPU layers)
music/          reference audio and MIDI transcription
scripts/        asset linking
```

## Documentation

- `PLAN.md` — plan, verified environment findings, acceptance criteria.
- `IMPLEMENTATION_LOG.md` — chronological discoveries, commands and measurements.
- `MIDI_ANALYSIS.md` — what the transcription contains and how it is used.
- `AUDIO_INVESTIGATION.md` — iteration-1 audio problems and the three-layer fix.
- `EXPERIMENTS.md` — controlled A–E comparison (baseline → full hybrid).
- `docs/architecture.md` — components, threading, data flow.
- `docs/mapping.md` — BPM → musical parameters → 25 Hz conditioning.
- `docs/commands.md` — CLI, scenarios, outputs, report fields.
- `docs/testing.md` — test strategy and how to reproduce the acceptance run.
- `docs/limitations.md` — honest limits of the prototype.
