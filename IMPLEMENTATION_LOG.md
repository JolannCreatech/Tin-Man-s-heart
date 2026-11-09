# DIASTOLE — Implementation Log

This is the project's persistent memory. Every significant discovery, command,
error and result is recorded here. Newest entries at the bottom.

---

## 2026-11-09 — Reconnaissance

**Repository:** `OpenCode_Diastole`, only `LICENSE` and `README.md`. Branch `Diastole`
(remote/local) contains an earlier non-functional sketch: sensor plot script,
`MIDI_factory.py`, a feature extractor, and a `MAG_prompter.py` that shelled out to
`mrt mlx generate` per prompt. Nothing was reused verbatim, but the two MIDI files
were copied into `music/` as prior reference transcriptions.

**Environment:**
- MacBook Pro `Mac14,7`, Apple M2, 8 GB, macOS 26.1, arm64.
- `uv 0.13.7`, system Python 3.9.6, target Python 3.12.14.
- Magenta clone: `~/Documents/magenta-realtime` (version 2.0.3, do not modify).
- Assets: `~/Documents/Magenta/magenta-rt-v2/`.
  - `models/mrt2_small/mrt2_small.mlxfn` (~435 MB)
  - `models/mrt2_small/mrt2_small_state.safetensors` (~8.3 MB)
  - `models/mrt2_base/` only has the state file (no `.mlxfn`).
  - `resources/musiccoca/*.tflite`, `resources/spectrostream/spectrostream_encoder.mlxfn`.
- Reference audio found at `~/Downloads/The Wizard Of Oz - If I Only Had A Heart.mp3`
  (170.5 s, 44.1 kHz stereo, 192 kbps). Copied to `music/`.
- Prior branch MIDI copied to `music/basic_pitch_transcription.mid`
  (format 1, 480 PPQ) and `music/major-scale.mid`.

**MLX compatibility (verified experimentally):**
```
mlx 0.32.4  -> RuntimeError: [import_function] Invalid string size
mlx 0.31.1  -> mx.import_function('.../mrt2_small.mlxfn') succeeds
```
Cause is a serialization-format incompatibility in MLX 0.32.x, not a corrupt model.
Decision: pin `mlx==0.31.1` in `pyproject.toml`.

**C++ standalone build:** blocked by missing Metal compiler:
`xcrun: error: unable to find utility "metal"`; `xcode-select -p` is
`/Library/Developer/CommandLineTools`, no full Xcode installed, sudo needs a password.
The Metal kernels cannot compile. Python + exported `.mlxfn` is the reliable path and
is on the critical path instead.

**MRT2 interface facts extracted from source (`magenta_rt/mlx/system.py`,
`export.py`, `depthformer.py`, `core/include/magentart/mlx_engine.h`):**
- 25 Hz frames, 1920 samples/frame, 48 kHz stereo; persistent streaming state.
- Real conditioning: 12 MusicCoCa tokens, 128-value onset pianoroll, 1 drum token,
  CFG scales. MIDI is not natively accepted.
- Exported `.mlxfn` argument order:
  `[cond, temperature, top_k, cfg_musiccoca, cfg_notes, cfg_drums,
    neg_musiccoca, neg_notes, forced_tokens, *state]`
- `forced_tokens` (shape `(1, 1, 12)`) allows audio prefill: feed SpectroStream RVQ
  tokens frame by frame to seed the transformer state. The C++ runtime uses this as
  `prefill_state`; the Python API does not expose it, so we implement it ourselves.
- SpectroStream encoder `.mlxfn` is stateless and traced for exactly 28 s of 48 kHz
  stereo audio (input padded/truncated accordingly).
- CFG notes scale is a strong adherence control (examples use 1.0–5.0).

Next action: scaffolding + Phase 1 MRT2 smoke test.

---

## 2026-11-09 — Scaffolding and environment

- Created `pyproject.toml`, `uv.lock`, `.python-version` (3.12), `.gitignore`,
  `scripts/link_assets.sh`, `src/diastole/`, `tests/`, `docs/`.
- `scripts/link_assets.sh` creates symlinks only:
  `assets/magenta-rt-v2/resources -> ~/Documents/Magenta/magenta-rt-v2/resources`
  and `.../models -> .../models`. No external files were modified or copied.
- Installed with `uv sync --extra dev`; confirmed Python 3.12 and
  `mlx==0.31.1`, `magenta-rt==2.0.3`.
- First MLX smoke test imported `mrt2_small.mlxfn` and generated frames from the
  exported function. Measured ~24–38 ms/frame on the M2; no reset between calls.

## 2026-11-09 — Core implementation

Implemented and wired:

- `sensors/` — `HeartRateSensor` ABC, `StaticSensor`, `SimulatedHeartRateSensor`
  with OU noise, RSA, scenarios (`rest`, `gradual`, `sudden`, `workout`, `demo`,
  `random`, `manual`) and manual override.
- `signal/` — validation, staleness, median, slew-limited EMA, arousal; mapping
  to tempo, density, note activity, dynamics, rhythmic intensity, drum gate and
  subdivision.
- `music/` — event timeline with tempo map, type-1 SMF writer (atomic replace),
  16th-note lookahead scheduler with a pitch profile from the reference MIDI,
  and the 25 Hz conditioning adapter.
- `generation/` — `MRT2Engine` around the exported `.mlxfn` (persistent 165-array
  state), SpectroStream prefill, cold-start priming, `MockEngine`.
- `audio/` — SPSC ring buffer, `sounddevice` playback thread with simulated sink
  fallback, underrun fades, WAV recorder.
- `runtime/` — orchestration, metrics, reports, graceful shutdown.
- `interface/` — Click CLI (`run`, `demo`, `prepare`, `inspect-reference`,
  `selftest`) and Rich live dashboard.

`uv run pytest -q` reached 42 passing tests before the final fixes.

## 2026-11-09 — Reference caches and prefill

Commands:

```bash
uv run diastole prepare
uv run diastole inspect-reference
```

Results:

- Reference audio 170.50 s at 48 kHz; MusicCoCa produced **35 style windows** of
  12 tokens (10 s clips, 5 s hop) -> `.cache/reference_style_e81cba618b547f7a.npz`.
- Pitch profile from `music/basic_pitch_transcription.mid`; top pitch classes
  `[3, 8, 10, 0, 7]`, register 48–76.
- SpectroStream encoded the 28 s excerpt (start 15 s) to **650 frames** in ~31 s.
- Manual prefill ran **649 forced frames** through the transformer. First attempt
  ~19.6 s; a later forced rebuild measured **26.4 s**. State is 165 arrays,
  8.3 MB, saved to `.cache/prefill_state_mrt2_small_6f83e8a26fd1947b.safetensors`.
  Cached load measured **0.01–0.03 s**.
- Forced-token encoding uses `raw + k*1024 + 6`, shape `(1,1,12)`; the last frame
  is injected into the previous-frame state slot (uint32). This reproduces the
  C++ runtime's `prefill_state` behaviour.

Cold vs prefilled comparison (earlier 6 s experiment): correlation 0.011, peak
0.816 vs 0.434 — the prefill changes the generated material substantially.

## 2026-11-09 — Real sessions and audio

- 6 s real model session with prefill: 27.7 ms/frame mean, 32.5 ms p95, 0
  underruns/drops, non-silent WAV/MIDI/report.
- 5 s real session with CoreAudio: `Audio sink: sounddevice`, playback succeeded.
- `selftest` and `selftest --full` pass; full test generated a non-silent frame at
  ~24 ms/frame.
- Mock pipeline runs produce valid WAV/MIDI/JSON and are used by tests.

## 2026-11-09 — Polish and correctness fixes

1. **Playback reporting bug:** `AudioPlayer.stop()` used to overwrite
   `mode="sounddevice"` with `"stopped"`, so a successful real session reported
   `audible=False`. Now `mode` keeps the backend, `was_audible` records whether a
   device was ever opened, and `active` records whether playback is running.
2. **Slew limit bug:** the 12 BPM/s limit was applied to the EMA *input*, which
   made the effective rise ~3.5 BPM/s and capped the demo at ~96 BPM. The limit
   now applies to the EMA *output*; the demo reaches ~133 BPM as intended. Also
   capped `dt` at 0.5 s to avoid a huge jump after a stall.
3. **Duplicate sensor samples:** the generation loop polls at 25 Hz while the
   sensor updates at 4 Hz. `BpmProcessor` now ignores repeated timestamps, so a
   frozen reading cannot dominate the median/EMA. Added a regression test.
4. **Prefill reporting:** reports now show the actual state-cache path, cache-hit
   flag, original compute seconds (from the JSON sidecar) and cache-load seconds.
   Previously the path was derived from a digest that did not match the loaded
   file and compute time was lost on cache hits.
5. **Generation gaps:** startup/pre-roll handoff gaps (~0.4 s) are reported
   separately and excluded from steady-state gap statistics; steady-state max was
   ~0.1 s.
6. **Audio validation:** reports now include clipped-sample count/fraction in
   addition to peak, RMS, NaN and silence. The final run had 0 clipped samples.
7. Added interpretable note/drum event rates overall and for high/low BPM
   segments, plus MIDI events per second.
8. **Interactive stop race:** with `--interactive`, the command reader called
   `pipeline.stop()` while the run loop's `finally` also called it. The second
   caller returned immediately (guard flag) and read the report before the first
   caller had written it, so the CLI printed "No report was produced." Added a
   `_stop_finished` event and an idempotent `stop()` that waits for the first
   shutdown to finish. Added a concurrent-stop regression test; the interactive
   `bpm 110 → status → stop` path now writes a complete report.

After fixes: `uv run pytest -q` -> **44 passed in ~3.7 s**.

## 2026-11-09 — Final acceptance run

```bash
uv run diastole demo --duration 42 --no-dashboard --out outputs/final_demo
```

Session `demo_20261109_222222` (real model, cached prefill, CoreAudio):

| Metric | Result |
|---|---|
| Audio | 42.04 s, 1051 frames, 48 kHz stereo WAV |
| Inference | 25.4 ms mean, 24.7 ms p50, 28.6 ms p95, 53.9 ms max |
| Real-time factor | 0.64 of the 40 ms budget |
| Buffer | 1.6 s capacity, 1.33 s mean fill, 0 underruns, 0 dropped |
| Playback | `sounddevice`, audible, 1,953,600 frames played |
| Audio level | peak 0.876, RMS 0.116, 0 clipped samples, no NaN |
| BPM | raw 60.4–137.0; smoothed 62.2–133.3 |
| Tempo | 70.4–116.7 |
| Onset rate | 2.05/s overall; 3.43/s high BPM vs 0.63/s low BPM |
| Drum-active frames | 1.14/s overall; 2.23/s high BPM vs 0.05/s low BPM |
| MIDI | 89 note events, 62 drum events, 106 tempo changes |
| Prefill | used, cache hit, 26.4 s original compute, 0.013 s load |
| Validation | non-silent, continuous, MIDI written, prefilled, audible all true |

Mock acceptance run (`--mock --no-audio --duration 42`) also completes with 0
underruns and shows the same qualitative BPM response. Outputs:
`outputs/mock_demo/`, `outputs/final_demo/`.

## 2026-11-09 — Manual BPM live acceptance

Ran a real-model, real-CoreAudio session and changed BPM from the controlling
thread while audio was playing (the same path as the interactive `bpm <n>`
command):

```python
p.start()
p.set_manual_bpm(62);  sleep(5)
p.set_manual_bpm(145); sleep(8)   # live command: bpm 145
p.set_manual_bpm(75);  sleep(7)   # live command: bpm 75
p.stop()
```

Output `outputs/manual_bpm/manual_bpm_live/`:

| Metric | Result |
|---|---|
| Audio | 21.36 s, 534 frames, `sounddevice`, audible |
| Inference | 25.5 ms mean, 27.3 ms p95 |
| Underruns / dropped | 0 / 0 |
| Smoothed BPM | 62 → 140.6 → 84 (raw up to 145) |
| Tempo | 70.3 → 121.4 → 84.6 |
| Density | 0.23 → 0.82 → 0.37 |
| Onset rate | 3.35/s high segment vs 0.85/s low segment |
| Drum-active frames | 2.23/s high vs 0.28/s low |
| Audio level | peak 0.855, RMS 0.087, 0 clipped |
| Validation | non-silent, continuous, MIDI written, prefilled, audible |

This confirms the live control path end to end, separate from the scripted demo
scenario. The smoothed BPM follows the 12 BPM/s slew limit rather than jumping.

## Reproduction

```bash
uv run pytest -q
uv run diastole selftest --full
uv run diastole demo --duration 42 --no-dashboard --out outputs/final_demo
```

Known limitations (simulated sensor only, ~1.3 s buffering latency, undrained
tail on stop, indirect model response, C++ build blocked) are documented in
`docs/limitations.md`.
