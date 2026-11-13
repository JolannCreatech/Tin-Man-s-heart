# Controlled experiments A–E

These experiments isolate what each iteration-2 layer contributes. They are
**not** a listening test: the agent can measure level, spectrum and timing but
cannot judge whether the music sounds good. Listening notes must be filled in by
a human using the WAVs listed below.

## Setup

| Item | Value |
|---|---|
| Command | `uv run diastole experiments --no-audio --duration 15 --bpm 70 --out outputs/experiments_fixed` |
| Model | real `mrt2_small` (not mock) |
| Duration | 15 s per run (15.04 s recorded) |
| Heart rate | fixed manual **70 BPM** → mapped tempo **75.5 BPM** |
| Seed | 7 |
| Audio device | none (`--no-audio`): paced simulated sink, WAV still records the mix |
| Reference cache | MIDI-render style windows + MIDI-render prefill, cached |

The first suite used the scripted `demo` scenario and was discarded as a
controlled comparison: each run sampled the BPM trajectory at a different wall
clock phase (means 64.7–117.0 BPM), confounding the layers. All numbers below
come from the fixed-BPM rerun.

Artifacts: `outputs/experiments_fixed/` (`summary.json`, one directory per run,
plus the rendered instrumental stem for D).

## Configurations

| Exp | Configuration | Mode | Heartbeat | MRT2 conditioning | MIDI accompaniment |
|---|---|---|---|---|---|
| **A** | Iteration-1 baseline | `legacy` | no | MP3 audio | no |
| **B** | MIDI analysis only | `midi` | no | none (MRT2 gain 0) | full |
| **C** | Heartbeat + current MRT2 | `legacy` | yes | MP3 audio | no |
| **D** | Instrumental MIDI conditioning | `legacy` | no | MIDI render + style text | no |
| **E** | Full hybrid | `hybrid` | yes | MIDI render + style text | full |

Gains: A/C/D `mrt2=0.52`; B `accompaniment=0.95, mrt2=0`; C/E
`heartbeat=0.62`; E `accompaniment=0.34, mrt2=0.50`; master 0.92. D and E use
text blend 0.55/0.65 respectively; A and C use audio style `drift`; B and D/E
use MIDI style `fixed`.

## Results

| Exp | Mode | Cond. | Heartbeat | BPM | Peak | RMS | Onsets/s | Clips | Underruns | Mean latency |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | legacy | audio | no | 70.0 | 0.603 | 0.051 | 0.73 | 0 | 0 | 26.3 ms |
| B | midi | — | no | 70.0 | 0.713 | 0.187 | 1.86 | 0 | 0 | 26.3 ms |
| C | legacy | audio | yes | 70.0 | 0.524 | 0.039 | 0.73 | 0 | 0 | 26.0 ms |
| D | legacy | midi | no | 70.0 | 0.237 | 0.035 | 0.73 | 0 | 0 | 27.1 ms |
| E | hybrid | midi | yes | 70.0 | 0.628 | 0.076 | 1.86 | 0 | 0 | 26.0 ms |

### Objective audio features

Computed from the recorded WAVs (STFT 2048/1024, mono):

| Exp | Spectral centroid | Low-band ratio (<200 Hz) | Spectral flatness | Heartbeat beats | Detected low pulse period |
|---|---:|---:|---:|---:|---:|
| A | 2852 Hz | 0.036 | 0.369 | 0 | — |
| B | 304 Hz | 0.407 | 0.002 | 0 | — |
| C | 2677 Hz | 0.136 | 0.363 | 19 | 0.7947 s |
| D | 1844 Hz | 0.021 | 0.168 | 0 | — |
| E | 692 Hz | 0.365 | 0.074 | 19 | 0.7947 s |

Expected beat period at the mapped tempo: `60 / 75.5 = 0.7947 s`.

### Heartbeat exactness

A separate heartbeat-only capture isolates Layer A end to end through the
playback thread and WAV recorder:

```bash
uv run diastole run --mock --no-audio --no-dashboard --no-reference --duration 10 \
  --bpm 72 --mode midi --mrt2-gain 0 --accompaniment-gain 0 \
  --heartbeat-gain 1.0 --master-gain 1.0 --out outputs/heartbeat_only
```

At raw BPM 72 the mapping gives musical tempo 76.8 BPM, so the expected period
is `60 / 76.8 = 0.78125 s`. Detected low-band thump intervals:

- 13 thumps, median **0.7812 s**, mean 0.7813 s, standard deviation
  **0.0001 s**, coefficient of variation **0.00013**.
- The same detection on C and E in the fixed suite gives a median interval of
  **0.7947 s**, exactly the expected mapped-tempo period.

There is no drift and no dependence on inference. In C/E the report records 19
heartbeat events over 15.04 s, consistent with ~18.9 beats at 75.5 BPM
(endpoint/phase effects explain the integer count).

## Interpretation

- **A (baseline)** is the iteration-1 sound: bright (centroid 2852 Hz), noisy
  (flatness 0.369), almost no low-frequency grounding (0.036), and no stable
  measured pulse. The model alone does not guarantee a beat.
- **B (MIDI material)** is the opposite extreme: a clean, tonal instrumental
  loop (flatness 0.002, centroid 304 Hz, strong low band). It proves the
  analysis/render path produces coherent musical material, but it is not
  adaptive and has no MRT2 colour.
- **C (heartbeat + MRT2)** adds an exact low pulse (low-band ratio rises from
  0.036 to 0.136; detected period exactly matches tempo) but the MRT2 layer
  remains bright/noisy (flatness 0.363). The pulse is fixed, the harmony is not.
- **D (MIDI conditioning)** changes MRT2's character substantially: centroid
  drops from 2677 to 1844 Hz and flatness from 0.363 to 0.168. It is quiet
  (peak 0.237) because no heartbeat or accompaniment is present; this is a gain
  choice, not silence.
- **E (full hybrid)** combines the exact pulse, the MIDI arrangement and the
  MIDI-conditioned MRT2. It has the lowest spectral flatness of the MRT2-bearing
  runs (0.074), a healthy low band (0.365), a good level (peak 0.628), zero
  clipping and zero underruns. This is the configuration the objective evidence
  supports.
- All runs had **0 clipped samples, 0 underruns and 0 dropped frames**. The
  limiter never engaged in these 15 s runs; gains alone kept the mix below full
  scale.

## Final real-device playback validation

The controlled suite used a simulated sink, so one additional session was run
through CoreAudio with the real model and the full hybrid mix:

```bash
uv run diastole run --no-dashboard --duration 20 --mode hybrid \
  --conditioning midi --style-mode fixed --text-blend 0.65 \
  --scenario manual --bpm 70 --out outputs/final_playback
```

| Metric | Result |
|---|---|
| Playback | `sounddevice` / **audible** |
| Audio | 20.04 s, peak 0.628, RMS 0.076 |
| Inference | 25.8 ms mean, 27.2 ms p95 (40 ms budget) |
| Underruns / dropped frames / voices | 0 / 0 / 0 |
| Heartbeat | 26 beat events, detected low-band period 0.7951 s (expected 0.7947 s at 75.5 BPM) |
| Clipped samples | 0 |
| Limiter reductions | 0 |

This confirms the device path, not musical quality: it played through the
system output while recording the same mixed signal. Subjective acceptance is
still the listening table below.

## Listening notes (to be completed by a human)

Files:

```
outputs/experiments_fixed/exp_A_baseline/audio.wav
outputs/experiments_fixed/exp_B_midi_material/audio.wav
outputs/experiments_fixed/exp_C_heartbeat_mrt2/audio.wav
outputs/experiments_fixed/exp_D_midi_conditioning/audio.wav
outputs/experiments_fixed/exp_E_full_hybrid/audio.wav
outputs/experiments_fixed/exp_D_midi_conditioning_midi_render.wav
```

Suggested comparison:

1. **Pulse**: does C/E feel like it has a steady heartbeat? Does A wander?
2. **Harmony**: does E sound grounded in Eb/Ab/Bb rather than drifting?
3. **Timbre**: does D/E sound more instrumental and less vocal-like than A/C?
4. **Balance**: is the heartbeat too loud/quiet relative to accompaniment and
   MRT2? Adjust `--heartbeat-gain`, `--accompaniment-gain`, `--mrt2-gain`.
5. **Chaos**: does E still contain obvious MRT2 artefacts or discontinuities?
6. **Musicality**: is the fixed 32 s loop too repetitive over 15 s? Over longer
   sessions?

| Exp | Pulse | Harmony | Timbre | Balance | Chaos | Notes |
|---|---|---|---|---|---|---|
| A | | | | | | |
| B | | | | | | |
| C | | | | | | |
| D | | | | | | |
| E | | | | | | |

## Caveats

- 15 s is enough to compare texture and timing, not enough to judge long-term
  repetition or fatigue.
- The runs used a simulated sink, not speakers. The mix is recorded at the same
  point the device would receive it, but acoustic latency and room sound were
  not evaluated.
- MRT2 sampling is seeded, but the harness does not guarantee bit-identical
  output across engine instances. Differences between A/C and D/E therefore
  combine the intended change with normal model variation.
- Objective features describe the signal, not musical quality. The listening
  table above is the missing evidence, and it requires a listener.
