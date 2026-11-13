# DIASTOLE — Implementation Plan

**Mission:** real-time physiological music generation with Magenta RealTime 2 (MRT2).
The system must continuously generate audio influenced by both (a) the user's heart
rate and (b) the original recording of *“If I Only Had a Heart”*.

**Status: iteration 1 complete and verified end-to-end** (Python/MLX path).
**Iteration 2** (MIDI-driven musical identity + deterministic heartbeat + mixer)
is implemented; controlled A–E experiments are in `EXPERIMENTS.md`. See §7 for
iteration-1 acceptance evidence, §8 for iteration 2, and `docs/limitations.md`
for what is not claimed.

Status legend: `[x]` done · `[~]` in progress · `[ ]` not started · `[!]` blocked.

---

## 1. Environment findings (verified, not assumed)

| Item | Finding |
|---|---|
| Repository | `OpenCode_Diastole`, initially only `LICENSE` + `README.md` |
| Machine | MacBook Pro `Mac14,7`, Apple M2, 8 GB RAM, macOS 26.1 arm64 |
| Python | system 3.9.6; `uv 0.13.7` available; target 3.12 |
| Magenta source | `~/Documents/magenta-realtime` (do not modify) |
| Magenta package | `magenta-rt` **2.0.3** in `~/Documents/magenta-realtime/.venv` |
| Model assets | `~/Documents/Magenta/magenta-rt-v2/models/mrt2_small/{mrt2_small.mlxfn,mrt2_small_state.safetensors}` (435 MB + 8.3 MB) |
| Resources | `.../resources/musiccoca/*.tflite`, `.../resources/spectrostream/{spectrostream_encoder.mlxfn,*.safetensors}` |
| MLX problem | `.mlxfn` import fails on MLX **0.32.4** with `RuntimeError: [import_function] Invalid string size`; **verified working on MLX 0.31.1** |
| C++ build | blocked: active tools are Command Line Tools only; `xcrun: unable to find utility "metal"`, no full Xcode, no passwordless sudo. Not on the critical path. |
| Real-time result | `mrt2_small` generated 42.0 s continuously at 25.4 ms/frame mean, 28.6 ms p95 (40 ms budget), 0 underruns, 0 dropped frames; `sounddevice`/CoreAudio output verified. |
| Prefill result | 650 SpectroStream tokens; 26.4 s one-time compute; 165-array state cached and loaded in ~0.013 s; output differs materially from a cold start. |
| Test result | `uv run pytest -q`: 44 passed in ~3 s (mock engine, no model load) after iteration 1; **54 passed** after iteration 2 (adds MIDI/mixer/synth/arranger coverage). |
| Reference music | `~/Downloads/The Wizard Of Oz - If I Only Had A Heart.mp3`, 170.5 s, 44.1 kHz stereo, 192 kbps |
| Reference MIDI | prior branch `Diastole`: `basic_pitch_transcription.mid` (format 1, 480 PPQ) and `major-scale.mid` |
| Audio output | `sounddevice` with CoreAudio works directly (MacBook Pro Speakers, device index 1); no Homebrew PortAudio package was needed. `afconvert`/`afplay` also exist. A paced simulated sink is used with `--no-audio` or when no device opens. |

### MRT2 capabilities that matter (from source, not guessed)

- Generation runs at **25 Hz**; each frame is **1920 samples = 40 ms**, 48 kHz stereo.
- Conditioning is **per frame**, and the transformer keeps a persistent streaming
  **state** between calls (KV cache / previous frame). This is the basis of continuity.
- Real conditioning inputs:
  1. `mulan_tokens_25hz` — 12 MusicCoCa style tokens (from text **or audio**),
  2. `pianoroll_with_onsets_tokens` — 128 values per frame
     (`-1` masked, `0` off, `1` sustaining, `2` onset, `3` free),
  3. `drum_pianoroll_tokens` — 1 value per frame (`-1` masked, `0` no drum, `1` drum),
  4. CFG scale tokens (musiccoca / notes / drums).
- Audio continuation is possible via **SpectroStream tokens**: the exported
  SpectroStream encoder turns 28 s of 48 kHz stereo audio into 25 Hz RVQ tokens;
  feeding those as `forced_tokens` through the exported transformer seeds its state.
  The C++ runtime implements exactly this as `prefill_state`.
- The **Python API does not expose prefill**, but the exported `.mlxfn` does accept
  `forced_tokens`; we will implement prefill ourselves in our engine wrapper.
- MIDI is **not accepted directly** by MRT2. MIDI is our canonical, inspectable
  control representation; an adapter converts the same event timeline into the
  25 Hz pianoroll/drum tensors that MRT2 actually consumes.

---

## 2. Target architecture

```
                 ┌─────────────────────────────────────────────┐
physiological    │ sensor connector (ABC)                      │
input ─────────▶ │   SimulatedHeartRateSensor (thread)         │
                 │   future: BLE / serial / OpenSignals        │
                 └───────────────┬─────────────────────────────┘
                                 │ PhysiologicalSample (bpm, ts, valid)
                                 ▼
                 ┌─────────────────────────────────────────────┐
                 │ signal processing: validate → median →      │
                 │ slew-limit → exponential smoothing           │
                 │ → PhysiologicalState                         │
                 └───────────────┬─────────────────────────────┘
                                 │ smoothed BPM, trend, arousal
                                 ▼
                 ┌─────────────────────────────────────────────┐
                 │ musical mapping (configurable, deterministic)│
                 │ tempo, density, energy, dynamics,            │
                 │ rhythmic intensity                          │
                 └───────────────┬─────────────────────────────┘
                                 │ MusicalParameters
                                 ▼
                 ┌─────────────────────────────────────────────┐
                 │ music/control                                │
                 │  • rolling MIDI timeline (tempo + notes +   │
                 │    percussion), written continuously         │
                 │  • 25 Hz adapter → pianoroll[128] + drum[1]  │
                 └───────────────┬─────────────────────────────┘
                                 │ per-frame conditioning tensors
                                 ▼
 original music ──▶ reference.py ──┬── MusicCoCa audio style tokens (evolving)
 *If I Only Had a Heart*           └── SpectroStream tokens → transformer prefill
                                 │
                                 ▼
                 ┌─────────────────────────────────────────────┐
                 │ generation/engine.py + magenta.py            │
                 │  persistent MRT2 streaming state, 1 frame    │
                 │  at a time, conditioning updated every frame │
                 └───────────────┬─────────────────────────────┘
                                 │ 1920×2 float32 @ 48 kHz
                                 ▼
                 ┌─────────────────────────────────────────────┐
                 │ audio/playback.py                            │
                 │  SPSC ring buffer → playback thread →        │
                 │  sounddevice/CoreAudio; session WAV recorder │
                 └───────────────┬─────────────────────────────┘
                                 ▼
                        speakers / headphones
                                 ▲
                 ┌───────────────┴─────────────────────────────┐
                 │ runtime/controller.py + interface/cli.py     │
                 │  lifecycle, live BPM, dashboard, metrics     │
                 └─────────────────────────────────────────────┘
```

### Why this split is musically coherent

- **BPM is not sent to MRT2 as “tempo”.** MRT2 has no tempo parameter and its clock is
  fixed at 25 Hz. BPM is therefore mapped to *event scheduling*: a tempo value
  determines the beat period, and rhythmic density/energy determine which 16th-note
  grid positions receive note onsets and drum hits. The resulting onset pattern is
  what reaches MRT2. Perceived tempo follows the pattern.
- **The MIDI file is not the conditioning input.** MRT2 cannot read MIDI. The MIDI
  file is the persistent, inspectable score; the same event objects are converted to
  the 25 Hz onset pianoroll/drum tensors the model consumes. There is a single source
  of truth (the event timeline), so file and model cannot drift apart.
- **The original song influences generation twice.** (1) Its audio is embedded with
  MusicCoCa and the style tokens evolve through the song’s structure over time.
  (2) A representative excerpt is encoded by SpectroStream and prefilled into the
  transformer state, so generation continues the actual musical context rather than
  merely being “prompted”. The raw recording is never mixed into the output.

### Threading / real-time model

- **Sensor thread** at 4 Hz (independent, never blocks inference).
- **Generation thread** at up to 25 frames/s; holds the MRT2 state; reads an immutable
  snapshot of the latest musical parameters and style tokens; writes frames to a ring
  buffer. It never touches the audio device.
- **Playback thread** pulls from the ring buffer and performs blocking
  `sounddevice.OutputStream.write()`; it never runs inference.
- **CLI/dashboard thread (main)** only reads metrics and writes control commands.
- Target buffer 1.6 s (85% fill target); underruns produce a short fade instead of a click.
- Graceful stop: stop generation, join threads, stop playback, drain and record the
  buffered tail, flush WAV/MIDI and reports.

---

## 3. Physiological → musical mapping (initial, configurable)

Inputs: raw BPM, filtered BPM `b`, rate of change `v` (BPM/s), variability `sd`,
arousal `a = smoothstep(50, 150, b) ∈ [0,1]`.

| Musical parameter | Mapping | Rationale |
|---|---|---|
| `tempo` (BPM) | `30 + 0.65·b`, clamped 50–180 | Keeps musical tempo in a usable range while preserving monotonic response (60→69, 100→95, 160→134). |
| `energy` | `a` | Overall intensity. |
| `rhythmic_density` | `0.20 + 0.65·a` | Fraction of 16th-grid positions that may carry an onset. |
| `note_activity` | `0.25 + 0.60·a` | Probability that an active step gets a pitched note. |
| `dynamics` | `0.45 + 0.50·a` | MIDI velocity; also the normalized target level. |
| `rhythmic_intensity` | `0.15 + 0.75·a` | Probability of drum hits on eligible grid positions. |
| `drum_gate` | 1 when intensity ≥ threshold else 0 | MRT2 drum token. |
| `beat_subdivision` | 2 at low arousal, 4 at high | Half-notes/quarters → 8ths/16ths. |

Additional rules:
- Note pitches come from a **pitch-class profile extracted from the reference MIDI
  transcription** (fallback: C major), voiced in a fixed register. This keeps the
  generated control material harmonically related to the song.
- Onsets are quantized to the current beat grid. A **lookahead scheduler** writes
  events for the next ~1 beat into a timeline; each 40 ms frame samples that timeline
  to produce onset/sustain/off tokens. This preserves temporal continuity and avoids
  “new random pattern every measurement”.
- Sudden BPM changes are slew-limited by the signal layer (default 12 BPM/s), so the
  grid tempo moves smoothly rather than jumping. The limit is applied to the
  filter output, after the EMA, so the configured rate is actually achieved.

---

## 4. Implementation phases (critical path first)

### Phase 0 — Project scaffolding and environment
- [x] `pyproject.toml` / `uv.lock`, Python 3.12 env, pinned `mlx==0.31.1`,
      `magenta-rt==2.0.3`, `sounddevice`, `mido`, `rich`, `pytest`.
- [x] `scripts/link_assets.sh` creating **symlinks** to existing model/resources (no
      duplicate downloads, no modification of external dirs).
- [x] `.gitignore`, `README.md`, `docs/` skeleton.
- **Verify:** `uv run python -c "import mlx.core as mx; mx.import_function(...)"` works.

### Phase 1 — MRT2 smoke test (CRITICAL PATH)
- [x] Load `mrt2_small.mlxfn`, generate 1 s with a text style prompt; save WAV.
- [x] Measure ms/frame; confirm < 40 ms for real-time on this M2.
- [x] Generate successive chunks using returned state; confirm continuity (no reset).
- [x] Inspect exported-function argument order and output dtype/shape.
- **Verify:** audible/non-silent WAV, state carried across calls, measured latency.

### Phase 2 — Reference music ingestion and audio prefill (CRITICAL PATH)
- [x] Load MP3 (soundfile or `afconvert` fallback), resample to 48 kHz stereo.
- [x] Precompute MusicCoCa audio embeddings for windows across the song; cache.
- [x] Encode a representative 28 s excerpt with the SpectroStream encoder.
- [x] Implement `prefill(tokens)` against the exported transformer (forced tokens);
      save/load resulting state to `.cache/` for fast restarts.
- [x] Verify that prefill changes the continuation’s character vs. a cold start.
- [x] Extract pitch-class profile from `basic_pitch_transcription.mid`.

### Phase 3 — Sensor + signal processing + mapping
- [x] Sensor ABC + `PhysiologicalSample` + status/connection model.
- [x] `SimulatedHeartRateSensor` (OU noise + respiratory sinus arrhythmia + scenarios:
      `rest`, `gradual`, `sudden`, `workout`, `manual`).
- [x] Processing: range validation, staleness, median filter, slew limiting, EMA.
- [x] Deterministic mapping to `MusicalParameters`.
- **Verify:** unit tests (values, manual override, outlier rejection, monotonic map).

### Phase 4 — MIDI timeline + 25 Hz conditioning adapter
- [x] Rolling event timeline (tempo map, note on/off, percussion) + SMF writer.
- [x] 25 Hz adapter producing `pianoroll[128]` and `drum[1]` tokens with correct
      `-1/0/1/2` semantics; sustain/onset tracking across frames.
- [x] Write a session MIDI file incrementally (not one file per measurement).
- **Verify:** parse with `mido`, check tempo events, note ranges, frame continuity.

### Phase 5 — Diastole generation engine
- [x] Custom streaming wrapper around the exported MLX function: build
      cond/neg/CFG args once per update, call one frame at a time, carry state.
- [x] Optional prefill state injection; reset; warm-up; state save/load.
- [x] Snapshot-based conditioning updates (no locks held during inference).
- [x] Mock engine for fast tests.
- **Verify:** 5 s generation, correct conditioning response, prefill round-trip.

### Phase 6 — Real-time audio playback
- [x] Use `sounddevice` with CoreAudio (no Homebrew PortAudio package needed here); simulated sink fallback.
- [x] SPSC ring buffer, playback thread, target fill, underrun handling/fade.
- [x] Session WAV recorder; clean shutdown.
- **Verify:** audible playback, measured buffer/latency, no gaps in 30 s run.

### Phase 7 — Runtime orchestration + CLI dashboard
- [x] `pipeline.py` wiring sensor → processing → mapping → control → engine → audio.
- [x] `cli.py`: `diastole run`, `diastole demo`, `diastole selftest`, live commands
      (`bpm 120`, `scenario sudden`, `status`, `stop`).
- [x] Rich dashboard: raw/smooth BPM, musical parameters, generation ms/frame,
      buffer fill, underruns, style window, reference, state.
- [x] Error handling and shutdown.

### Phase 8 — Validation and documentation
- [x] Full test suite (`uv run pytest`).
- [x] End-to-end acceptance run with BPM changes; save WAV, MIDI, JSON report.
- [x] Analyze report: BPM responsiveness (onset/drum density in high vs low BPM
      segments), non-silence, continuity, latency.
- [x] Update `README.md`, `PLAN.md`, `IMPLEMENTATION_LOG.md`, `docs/*`.
- [x] Record remaining limitations honestly.

---

## 5. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MLX version incompatibility | **confirmed** | Pin `mlx==0.31.1`; verified `import_function` works. |
| Model too slow for 25 Hz on M2 | medium | Use `mrt2_small`, 8-bit if needed; measure; buffer absorbs jitter; document. |
| Audio device unavailable | low | Try `sounddevice` at start; fall back to a paced simulated sink, keep recording, and report `audible_output=false`. |
| Python prefill too slow / unsupported | medium | Prefill 12–20 s once, cache state `.safetensors`; fallback to style-only conditioning (still genuine reference influence). |
| Conditioning updates destabilize audio | medium | Update note/drum tokens every frame, style every 1 s; maintain state; smooth parameters; test. |
| MP3 decode unavailable | low | `afconvert` to 48 kHz WAV cache inside repo. |
| 8 GB RAM pressure | low | Small model only; free state arrays promptly; monitor RSS. |

## 6. Acceptance criteria (Definition of Done)

1. `uv run diastole run` (documented) starts the simulated sensor, loads the local
   MRT2 small model, loads the reference song, and plays continuous audio.
2. Live `bpm <n>` changes the processed parameters and subsequent generation; this is
   visible in the dashboard and measurable in the recorded WAV/MIDI.
3. The reference song genuinely conditions generation (MusicCoCa style tokens +
   SpectroStream prefill), not merely played on top.
4. A single session produces a WAV and a MIDI file covering the whole run.
5. Measured generation latency and buffer metrics are reported; no obvious clicks or
   gaps in a 30 s run under normal conditions.
6. Graceful `stop`/Ctrl-C; all threads join; files flushed.
7. `uv run pytest` passes (unit + non-slow integration); slow model tests documented.
8. Documentation explains architecture, mapping, commands, and limitations.

---

## 7. Final status

**All phases complete on the Python/MLX path.** The prototype runs end-to-end:
simulated heart rate → processing → mapping → timeline/MIDI → 25 Hz adapter →
prefilled MRT2 stream → ring buffer → CoreAudio, with a WAV, MIDI and JSON/Markdown
report per session.

Acceptance evidence (final 42 s run, `outputs/final_demo/`):

1. `uv run diastole demo --duration 42` loaded the local `mrt2_small` model and the
   reference song, opened `sounddevice`, and played continuously.
2. Live/manual BPM control was verified in a real-model, real-audio session that
   sent `set_manual_bpm(145)` then `75` during playback: smoothed BPM moved
   62 → 140.6 → 84, tempo 70 → 121 → 85, onset rate 3.35 vs 0.85/s and
   drum-active frames 2.23 vs 0.28/s (high vs low segment). The scripted demo
   independently produced 3.43 vs 0.63 onsets/s and 2.23 vs 0.05 drum-active
   frames/s.
3. Reference conditioning is genuine: 35 MusicCoCa windows and a 650-frame
   SpectroStream prefill; cold vs prefilled output differed materially in an
   earlier comparison.
4. One WAV (42.04 s) and one MIDI file cover the session.
5. Inference averaged 25.4 ms/frame (p95 28.6 ms) against a 40 ms budget, with
   0 underruns and 0 dropped frames. One outlier frame took 53.9 ms; the 1.6 s
   buffer absorbed it.
6. Ctrl-C/`stop` path joins threads and flushes artifacts. (Iteration 2 also
   drains and records the buffered tail; see §8.)
7. `uv run pytest -q` → 44 passed in iteration 1, 54 in iteration 2.
8. Documentation: `README.md`, this plan, `IMPLEMENTATION_LOG.md` and `docs/`.

Remaining blocked item: the optional C++ Magenta runtime build (no Metal
compiler). It is not required for the delivered prototype.

Remaining limitations are recorded in `docs/limitations.md`; the main ones are
the simulated-only sensor, indirect/non-guaranteed musical response, ~1.3 s
playback latency from buffering, and (in iteration 1) the undrained tail on
stop. Iteration 2 drains the tail and adds a limiter; iteration-2 limitations
are listed in §8 and `docs/limitations.md`.

---

## 8. Iteration 2 — MIDI identity, deterministic pulse, layered mix

**Goal:** keep the working prototype but replace the chaotic, vocal-leaning
MRT2 output with a coherent arrangement built from three separable layers:

1. **Layer A — heartbeat:** BPM-exact, rendered in the playback thread, stable
   even when inference stalls.
2. **Layer B — MRT2:** reduced to complementary harmonic/textural material,
   conditioned on an instrumental MIDI render rather than the vocal MP3.
3. **Layer C — MIDI accompaniment:** bass/pad/lead derived from the
   transcription, schedule-locked to the live BPM.

The MIDI file is the primary musical reference; the MP3 is comparison-only and
is never mixed into the output.

### Phase 9 — MIDI analysis and material extraction
- [x] Parse SMF, tempo map (fixed end-tick inversion), time signature, programs.
- [x] Key estimation (Krumhansl), bass-anchored diatonic Viterbi chords, bars.
- [x] Automatic 16-bar loop selection (bars 44–59) and cleaned theme.
- [x] Diagnostics + `midi_analysis.json` per session.
- [x] Unit tests on the real transcription.

### Phase 10 — Offline instrumental rendering
- [x] Self-contained numpy synth (bass/pad/lead), no FluidSynth/SoundFont.
- [x] Deterministic render + cache (`midi_render_*.wav`).
- [x] Loop material shared with the live arranger.

### Phase 11 — BPM-locked arranger and real-time layers
- [x] `MidiArranger` walks the beat clock and emits sample-accurate events.
- [x] `AccompanimentSynth` (pending pre-rendered voices).
- [x] `HeartbeatSynth` rendered in the playback thread from live BPM.
- [x] Heartbeat also available in `legacy` mode for the C ablation.
- [x] Unit tests for beat integration and heartbeat exactness.

### Phase 12 — Mixer, playback and recording
- [x] `SmoothedGain`, instant-attack `Limiter`, `LayerMixer` with per-layer
      gains, optional ducking, optional reference layer, clip guard.
- [x] Playback-thread processor + recorder; periodic flush; drain on stop.
- [x] WAV now captures the actual mixed output.
- [x] Unit tests for gains, limiter and reference routing.

### Phase 13 — Conditioning on the MIDI render
- [x] `ReferenceConfig.conditioning_source` (`midi` default, `audio` ablation).
- [x] MusicCoCa style windows from the instrumental render.
- [x] SpectroStream prefill from the render.
- [x] Text/audio style blend before tokenisation (fixed float32 dtype bug).
- [x] Style modes `fixed` / `drift` / `loop`.

### Phase 14 — CLI, experiments and documentation
- [x] CLI modes `legacy|midi|hybrid`, per-layer gains, conditioning options.
- [x] `diastole experiments` runner for A–E with audio/report/metrics/stems.
- [~] Run controlled A–E experiments and record listening notes.
- [x] `MIDI_ANALYSIS.md`, `AUDIO_INVESTIGATION.md`, `EXPERIMENTS.md`.
- [x] Update README, limitations and implementation log.

### Iteration-2 acceptance criteria
1. A real-MRT2 hybrid session is continuous, non-silent, has 0 underruns and
   0 clipped samples, and records the mixed output. **Evidence:** 20 s audible
   `sounddevice` run in `outputs/final_playback/`, peak 0.628, RMS 0.076,
   0 underruns, 0 clipped.
2. The heartbeat layer advances independently of inference at the live BPM.
   **Evidence:** heartbeat-only capture median interval 0.7812 s vs expected
   0.78125 s (CV 0.00013); final run 26 beat events, 0.7951 s vs 0.7947 s.
3. MIDI-derived accompaniment is BPM-locked and uses the analysed loop.
   **Evidence:** `MidiArranger` tests at 60/120 BPM; B/E runs at fixed 70 BPM
   with 1.86 onsets/s and 10 style windows from the MIDI render.
4. MRT2 is conditioned on the instrumental MIDI render (style + prefill) by
   default and no longer receives the lead melody or drum pulse.
   **Evidence:** 10 style windows and MIDI-render prefill in D/E; conditioning
   timeline reduced to bass + pad.
5. A–E runs are saved with comparable metrics and representative audio.
   **Evidence:** `outputs/experiments_fixed/summary.json` and `EXPERIMENTS.md`.
6. Tests cover every new component and the full suite passes.
   **Evidence:** `uv run pytest -q` → 54 passed.
7. Limitations are updated honestly, including the inability of the agent to
   judge musical quality without a listener. **Evidence:**
   `docs/limitations.md`; the listening table in `EXPERIMENTS.md` is left for
   a human.
