# Architecture

DIASTOLE has one job: keep a single MRT2 stream alive while physiological and
reference-song signals change the **conditioning** of that stream. It never
restarts the model per BPM measurement.

## Data flow

```
HeartRateSensor (4 Hz thread)
    -> PhysiologicalSample(bpm, timestamp, valid)
BpmProcessor
    -> validate -> median -> slew-limited EMA -> PhysiologicalState
BpmMapper
    -> MusicalParameters(tempo, density, activity, dynamics, intensity, ...)
┌────────────────────────── generation thread ──────────────────────────┐
│ MusicalTimeline (full, written to MIDI)                               │
│   hybrid: MidiArranger walks the beat clock -> bass/pad/lead events   │
│   legacy: TimelineScheduler on a 16th-note grid                       │
│ MRT2 conditioning timeline (hybrid: bass + pad only)                  │
│   -> FrameAdapter -> style[12] + notes[128] + drum[1] at 25 Hz        │
│ MRT2Engine (persistent streaming state)                               │
│   -> 1920x2 float32 @ 48 kHz per frame -> RingBuffer                  │
└───────────────────────────────────────────────────────────────────────┘
RingBuffer -> AudioPlayer thread
    HeartbeatSynth (live BPM) + AccompanimentSynth (pre-rendered voices)
    -> LayerMixer (per-layer gains, ducking, limiter, clip guard)
    -> CoreAudio
    -> WavRecorder -> audio.wav (actual mixed output)
```

`MusicalTimeline` is the single source of truth for the full arrangement. The
MIDI writer and the 25 Hz adapter read the same event objects, so the score on
disk and the tensors sent to the model cannot drift apart. In `hybrid` mode a
second, reduced conditioning timeline carries only bass and pad notes; the lead
theme and the heartbeat pulse are deliberately kept away from MRT2.

## Three layers

| Layer | Implementation | Thread | Depends on inference? |
|---|---|---|---|
| A. Heartbeat | `audio/synth.py::HeartbeatSynth` | playback | no |
| B. MRT2 | `generation/engine.py` | generation | yes |
| C. MIDI arrangement | `music/arranger.py` + `audio/synth.py::AccompanimentSynth` | generation schedules; playback renders | no (the playback thread keeps adding pre-rendered voices) |

The heartbeat advances an absolute beat counter from the current BPM, so it
stays exact across inference stalls and tempo changes.

## Threading and real-time behaviour

| Thread | Rate | Responsibility |
|---|---|---|
| sensor | 4 Hz | Simulated acquisition; timestamps each reading. |
| generation | up to 25 Hz | Owns the MRT2 state, arranger/scheduler, synth voice scheduling, metrics and MIDI snapshots. |
| playback | 20 ms blocks | Drains the ring buffer, renders heartbeat + accompaniment, mixes/limits, writes the device and the WAV. Never runs inference. |
| CLI/dashboard | 5 Hz | Reads immutable snapshots, sends commands. |

- Ring buffer capacity: **1.6 s**. Target fill 85%; pre-roll fills at least
  0.75 s (19 frames at 40 ms) before the playback thread starts.
- The generation loop yields with a short wait when the buffer is full and does
  not dump a burst of past notes after a stall.
- Underruns are faded in/out over 64 samples instead of producing a hard click.
- A slow first inference cannot create an audible startup gap because playback
  starts only after pre-roll.

`AudioPlayer.mode` records the backend selected for the session
(`sounddevice` or `simulated`). It is not overwritten on stop, and
`was_audible` records whether a real output device was ever opened, so reports
describe how the session played rather than only how it ended.

## MRT2 integration

`generation/engine.py` wraps the exported `.mlxfn` directly through MLX:

- Arguments: `[cond, temperature, top_k, cfg_musiccoca, cfg_notes, cfg_drums,
  neg_musiccoca, neg_notes, forced_tokens, *state]`.
- Conditioning width 141 = 12 MusicCoCa style tokens + 128 note tokens + 1 drum
  token. Raw mask is `-1`; the export offset is `+7`.
- Each call returns audio plus the next 165 state arrays; the engine keeps them,
  so generation is one continuous stream.
- The cold model emits a silent first frame. When no prefill is installed the
  pipeline performs one fully masked priming step and discards that frame.
- `mrt2_small` is used because it fits 8 GB and runs well inside the 40 ms frame
  budget. `MockEngine` provides a deterministic model-free path for tests.

### SpectroStream prefill

MRT2's Python API does not expose audio continuation, but the exported function
accepts `forced_tokens`. The engine implements the same mechanism as the C++
runtime's `prefill_state`:

1. Encode a 28 s excerpt of the conditioning source (by default the MIDI
   render; with `--conditioning audio`, the MP3 from 15 s) with
   `spectrostream_encoder.mlxfn` at 25 Hz -> `(650, 12)` RVQ codes.
2. Make each forced code unique across the 12 codebooks with
   `raw + k*1024 + 6`, shape `(1, 1, 12)`.
3. Run 649 forced frames through the transformer, then inject the last frame
   into the state slot that represents the previous frame.
4. Cache the 165-array state as `.cache/prefill_state_<model>_<digest>.safetensors`
   with a JSON sidecar recording the original compute time.

A cold start and a prefilled start produce materially different audio (earlier
comparison: correlation 0.011, peak 0.816 vs 0.434), confirming that the
recording changes generation rather than merely playing alongside it.

### Reference conditioning

- By default the conditioning source is an **instrumental rendering of the MIDI
  transcription**, not the MP3 (`ReferenceConfig.conditioning_source="midi"`).
  The 4/4 120 BPM transcription is analysed, a 16-bar loop is selected
  (bars 44–59, Eb–Ab–Bb–Eb), and a 48 s deterministic bass/pad/lead render is
  cached under `.cache/reference/`.
- MusicCoCa embeds 10 s windows with a 5 s hop across that render, producing 10
  style windows of 12 RVQ tokens; the active window advances with session time
  (`fixed`, `drift` or `loop` modes). A text embedding can be blended in before
  tokenisation (`--text-style`, `--text-blend`) so the model is nudged toward
  "instrumental, no vocals".
- The prefill excerpt is taken from the same render, so the transformer state
  continues instrumental material rather than a vocal recording.
- The MP3 remains available for the `--conditioning audio` ablation and is
  still analysed for comparison; it is never mixed into the output.
- The pitch-class profile for the legacy note scheduler comes from
  `music/basic_pitch_transcription.mid`, voiced in MIDI register 48–76.

## Session artifacts

Every session directory contains:

| File | Contents |
|---|---|
| `audio.wav` | 16-bit stereo 48 kHz recording of the **mixed** output (playback thread). |
| `score.mid` | Type-1 SMF with tempo map, notes and drums; atomically rewritten every 2 audio seconds and at shutdown. |
| `midi_analysis.json` | Key, chords, bars, loop, theme, diagnostics and stats (non-legacy modes). |
| `report.json` | Configuration, measurements, validation and per-frame time series. |
| `report.md` | Human-readable summary of the same report. |

## Shutdown

`Pipeline.stop()` stops the generation loop, joins it, stops playback (draining
the ring and recording the tail because `AudioConfig.drain_on_stop=True`), stops
the sensor, closes the WAV, writes the final MIDI and both reports. Ctrl-C
follows the same path.
