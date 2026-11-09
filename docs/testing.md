# Testing and reproduction

## Unit / integration suite

```bash
uv run pytest -q          # 44 tests, ~3 s, no model load
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
| `test_pipeline.py` | 3 | end-to-end mock session (WAV/MIDI/JSON), manual BPM changes parameters, concurrent `stop()` produces one complete report. |

The suite includes a real 1.2 s mock pipeline session that asserts continuous
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
```

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

## Timing note

The host has no GNU `timeout` command. Long real-model commands are run with the
agent/tool timeout or in the background; internal durations are reported by the
session itself (`session.wall_seconds`, `audio_seconds`, `latency_ms`).
