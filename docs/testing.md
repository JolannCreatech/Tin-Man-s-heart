# Testing and reproduction

## Unit / integration suite

```bash
uv run pytest -q          # 54 tests, ~5 s, no model load
uv run pytest --cov       # optional coverage
uv run pytest tests/test_processing.py -q
```

The default suite uses `-m 'not integration'` (configured in `pyproject.toml`),
so it never loads the 435 MB model or touches an audio device.

| File | Tests | Coverage |
|---|---:|---|
| `test_processing.py` | 9 | range/outlier rejection, median spike suppression, staleness, duplicate timestamps, slew limit, trend/variability, reset/fallback. |
| `test_mapping.py` | 4 | monotonic BPM→parameters, bounds, drum gate, subdivision. |
| `test_sensors.py` | 5 | scenarios, manual override, reset clock, status, determinism. |
| `test_timeline.py` | 7 | notes/drums, tempo map, tick conversion, atomic MIDI, scheduler continuity, pitch constraints. |
| `test_adapter.py` | 4 | note on/off/sustain/onset tokens, drum token, style validation. |
| `test_ring_buffer.py` | 5 | wraparound, capacity, underruns, clear, fill. |
| `test_engine.py` | 3 | conditioning layout/offset/masks, mock engine determinism. |
| `test_reference.py` | 4 | cache index/digest determinism, style window selection, pitch profile fallback. |
| `test_pipeline.py` | 4 | end-to-end mock session (WAV/MIDI/JSON), manual BPM changes parameters, concurrent `stop()` produces one complete report, hybrid mode renders/records the mix. |
| `test_music_layers.py` | 9 | real MIDI analysis, deterministic render/cache, voice shapes, beat-locked arranger (60/120 BPM), BPM-exact heartbeat, accompaniment scheduling, mixer gains/limiter/reference, gain smoothing. |

The suite includes real 1.2 s mock pipeline sessions that assert continuous
frames, non-silence, exact WAV length, MIDI presence and report contents.

## Real-model checks

```bash
uv run diastole selftest          # assets, MLX import, devices, WAV
uv run diastole selftest --full   # + 3 real MRT2 frames, prints peak and ms/frame
```

Expected: all checks PASS; `MRT2 streaming step` peak well above `1e-4` and
around 24–30 ms/frame on an M2.

## Reproduce the acceptance run

```bash
# optional, only if .cache/ is empty
uv run diastole prepare

# model-free pipeline sanity check
uv run diastole demo --mock --no-audio --no-dashboard --duration 42 --out outputs/mock_demo

# real model + real audio output
uv run diastole demo --duration 42 --no-dashboard --out outputs/final_demo
```

The real run prints the selected audio sink. On this Mac it is `sounddevice`
(CoreAudio). If no device is available it falls back to a paced simulated sink;
the WAV and buffer metrics are still produced, but `playback.audible` is false.

### Live manual BPM check

The interactive `bpm <n>` command and `Pipeline.set_manual_bpm()` use the same
path. A scriptable real-model check:

```python
from pathlib import Path
import time
from diastole.config import RuntimeConfig
from diastole.runtime.pipeline import Pipeline

p = Pipeline(
    RuntimeConfig(duration=22.0, scenario="rest", output_dir=Path("outputs/manual_bpm")),
    use_reference=True,
    session_name="manual_bpm_live",
    progress=print,
)
p.set_manual_bpm(62); p.start()
time.sleep(5); p.set_manual_bpm(145)
time.sleep(8); p.set_manual_bpm(75)
time.sleep(7); p.stop()
```

The resulting report should show smoothed BPM rising toward 145, then falling
toward 75 under the 12 BPM/s slew limit, with higher onset/drum rates in the
high-BPM segment. The verified run is recorded in `IMPLEMENTATION_LOG.md`.

### What to verify in the report

Open the newest `outputs/final_demo/*/report.json`:

```python
import json, glob
r = json.load(open(sorted(glob.glob("outputs/final_demo/*/report.json"))[-1]))
print(r["latency_ms"])       # mean ~25 ms, p95 < 30 ms, budget 40 ms
print(r["buffer"])           # underruns 0, dropped 0
print(r["audio"])            # peak > 0.01, no NaN, clipped 0
print(r["validation"])       # all true
print(r["musical"])          # high-BPM onset/drum rates > low-BPM rates
print(r["prefill"])          # used, cache_hit, compute/load seconds
print(r["arrangement"])      # mode, key, loop, chords, mixer layer stats
```

### Iteration-2 checks

```bash
# MIDI material only (no MRT2, no heartbeat)
uv run diastole run --mock --no-audio --no-dashboard --duration 5 \
  --mode midi --mrt2-gain 0 --no-heartbeat --out outputs/midi_only

# Heartbeat-only capture at a fixed BPM; verifies Layer A timing
uv run diastole run --mock --no-audio --no-dashboard --no-reference --duration 10 \
  --bpm 72 --mode midi --mrt2-gain 0 --accompaniment-gain 0 \
  --heartbeat-gain 1.0 --master-gain 1.0 --out outputs/heartbeat_only

# Controlled A-E suite
uv run diastole experiments --duration 15 --bpm 70 --out outputs/experiments
```

For the heartbeat-only capture, detect low-band peaks and check that intervals
equal `60 / mapped_tempo`. At raw BPM 72 the mapping gives tempo 76.8, so the
expected period is 0.78125 s. The verified capture measured a median interval of
**0.7812 s**, standard deviation **0.0001 s** (CV 0.00013) across 13 thumps —
i.e. sample-accurate and independent of inference. See `EXPERIMENTS.md`.

Acceptance thresholds used here:

| Check | Threshold |
|---|---|
| Continuous audio | 1051 frames for 42.04 s at 25 Hz |
| Real-time | mean inference < 40 ms/frame, 0 underruns, 0 dropped |
| Audible | `playback.mode == "sounddevice"` and `validation.audible_output` |
| Non-silent | peak > 0.01, no NaN, no clipped samples |
| Musical response | high-BPM onset and drum rates clearly exceed low-BPM rates |
| Continuity | MIDI note/drum events span the session; one WAV/MIDI per session |
| Reference | style windows > 0 and `prefill.used` true |
| Heartbeat | `heartbeat_beat_events` > 0; offline period matches mapped tempo |
| MIDI material | `arrangement.key`, loop and chord list present; render non-silent |
| Mix safety | 0 clipped samples; limiter is a final guard, not a substitute for gains |

## Timing note

The host has no GNU `timeout` command. Long real-model commands are run with the
agent/tool timeout or in the background; internal durations are reported by the
session itself (`session.wall_seconds`, `audio_seconds`, `latency_ms`).
