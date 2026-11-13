# Audio investigation and three-layer redesign

This document records why the iteration-1 output sounded chaotic and how the
iteration-2 audio path addresses it. It is an engineering investigation, not a
claim that the result is musically finished. Subjective judgement still requires
a listener.

## Symptoms in iteration 1

Listening notes and metrics from the first prototype converged on four problems:

1. **No dependable pulse.** The heartbeat only existed as a *probability* that
   MRT2 might produce a drum token (`rhythmic_intensity`, `drum_gate`). The model
   could ignore it for seconds at a time, especially at low CFG or high
   temperature, so perceived tempo wandered even when the mapped BPM was stable.
2. **Vocal-like / unstable timbre.** MusicCoCa style windows and the
   SpectroStream prefill were taken from the original MP3, which is a
   vocal-forward 1939 recording. The model was therefore encouraged to emit
   voice-like artefacts that no note-level conditioning could discipline.
3. **Harmonic drift.** The iteration-1 scheduler picked pitches from a global
   pitch-class profile. It did not follow the song's chord changes, so the
   harmonic centre drifted and the generated texture had no functional bass.
4. **Buffer stalls were audible as musical instability.** Inference occasionally
   exceeded the 40 ms frame budget. Any layer rendered inside the generation
   thread stops with it; only layers rendered in the playback thread survive.

The first three are musical; the fourth is architectural. The redesign treats
them separately.

## Architecture: three independent layers

```
                 ┌────────────────────────────────────────────┐
 generation      │ MIDI analysis → 16-bar loop → arranger     │
 thread          │   deterministic bass / pad / lead events   │
                 │   sample-accurate against the BPM clock    │
                 └───────────────┬────────────────────────────┘
                                 │ pre-rendered voices
                                 ▼
                 ┌────────────────────────────────────────────┐
 playback        │ HeartbeatSynth (BPM-exact, playback thread)│
 thread          │ AccompanimentSynth (scheduled voices)      │
                 │ MRT2 block from ring buffer                │
                 │ → LayerMixer → Limiter → clip → device/WAV │
                 └────────────────────────────────────────────┘
                                 ▲
 generation      ┌───────────────┴────────────────────────────┐
 thread          │ MRT2 (Layer B): harmonic/textural colour   │
                 │ conditioned on MIDI-derived material       │
                 └────────────────────────────────────────────┘
```

### Layer A — deterministic heartbeat

- `src/diastole/audio/synth.py::HeartbeatSynth` renders a lub-dub pattern plus
  optional subdivision ticks from the *current BPM*, inside the playback thread.
- It advances an absolute musical beat counter by exactly
  `frames / (60/BPM · sample_rate)` per block, so tempo changes alter the future
  beat period without retriggering or drifting.
- It does not read MRT2 state, the ring buffer, or the sensor. If inference
  stalls, the heartbeat keeps beating.
- Pattern, subdivision and accents are configurable (`--heartbeat-pattern`,
  `--no-heartbeat`, `ArrangementConfig`).

### Layer B — MRT2 accompaniment

MRT2 now receives a deliberately *reduced* conditioning timeline:

- Only bass and pad notes from the MIDI-derived arrangement reach the model.
- The lead theme is **not** sent to MRT2; it is rendered deterministically by
  Layer C. This prevents the model from doubling or corrupting the melody.
- The drum token stays `0`; the model is not asked to invent a pulse that
  Layer A already owns.
- The style/prefill source defaults to the instrumental MIDI render, not the
  vocal MP3 (`--conditioning midi`).
- A MusicCoCa text embedding is blended with the audio style embedding before
  tokenisation (`--text-style`, `--text-blend`).

### Layer C — MIDI-derived accompaniment

`src/diastole/music/arranger.py` walks the live beat clock. Each 40 ms
generation frame advances the beat by `tempo · 0.04 / 60`, and events that fall
inside the frame are placed at their exact sub-frame sample offset. Voices are
pre-rendered by `src/diastole/music/midi_render.py` and pushed to
`AccompanimentSynth`, so the playback thread only mixes buffers.

The rendered loop can be auditioned in isolation
(`load_or_render(...)` / cached `midi_render_*.wav`) and is the musical
reference the other layers follow.

### Mixer and safety

`src/diastole/audio/mixer.py::LayerMixer` applies:

- independent, smoothed gains per layer (`heartbeat`, `accompaniment`, `mrt2`,
  optional `reference`);
- optional sidechain ducking of MRT2/accompaniment by the heartbeat;
- a master gain;
- an instant-attack / smooth-release limiter followed by a hard clip guard.

Playback and recording happen in the playback thread
(`AudioPlayer(processor=..., recorder=...)`). `drain_on_stop=True` plays and
records the buffered tail so the WAV is the actual mixed output rather than the
raw model stream.

## Measurements

### Offline material

- MIDI analysis: Eb major, 761 notes, 71 chord segments, 16-bar loop at bars
  44–59.
- Instrumental render of the 32 s loop: peak 0.89, RMS 0.21, produced in
  ~0.15 s for 20 s of audio. Deterministic across runs.

### Real MRT2 hybrid session (12 s, `outputs/real_hybrid_check2/`)

| Metric | Result |
|---|---|
| Conditioning source | instrumental MIDI render |
| MusicCoCa style windows | 10 (text blend 0.65) |
| Prefill | 650-frame MIDI-render excerpt, cached load |
| Inference | 25.5 ms mean, 27.5 ms p95 (40 ms budget) |
| Underruns / dropped | 0 / 0 |
| Audio | 12.04 s, 48 kHz stereo |
| Peak / RMS | 0.654 / 0.077 |
| Clipped samples | 0 |
| Low-band events | 13 over 12 s at ~62 BPM (matches the beat count) |
| Mode / heartbeat | hybrid / active |

The heartbeat is not proven by this low-band count alone because the bass also
lives below 140 Hz. The deterministic synth is covered directly by
`tests/test_music_layers.py` (60 BPM → 2 beats in 2 s, 120 BPM → 4 beats), and
the pipeline report exposes `heartbeat_beat_events`, which increments
independently of inference.

### Bugs found during the investigation

1. **Tempo-map inversion.** The beat converter only knew tempo events at tick 0,
   so the whole song collapsed to 4 beats and the loop selector saw one chord.
   Fixed by passing the MIDI end tick to `_TempoConverter`.
2. **MusicCoCa tokeniser dtype.** Blending text and audio embeddings produced
   float64 arrays; the tflite tokeniser expects float32 and rejected them, so
   style conditioning silently fell back to masked tokens. Fixed by casting the
   final embedding to float32. The next real run reported 10 style windows.
3. **Recording the wrong signal.** In iteration 1 the WAV was written from the
   generation thread before the mix. Recording now happens after the playback
   thread's processor, so the captured WAV is what was played.
4. **Undrained tail.** `drain_on_stop` was reserved but unused, leaving ~1.3 s
   of buffered audio unheard/unrecorded. It now drains and records on stop.

## What this does and does not fix

It fixes the *pulse*, *harmonic grounding* and *conditioning source* problems by
construction: those layers are deterministic and cannot be overridden by the
model. It reduces, but cannot eliminate, model timbre drift; MRT2 still colours
the output and can still produce artefacts. The controlled A–E experiments are
the evidence for how much each change contributes; see `EXPERIMENTS.md`.
