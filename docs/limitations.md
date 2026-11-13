# Limitations (honest list)

DIASTOLE is a working prototype, not a finished instrument. These are the real
limits of the current implementation.

## Sensing

- Only the **simulated** heart-rate source is implemented. The `HeartRateSensor`
  ABC is ready for BLE/serial/OpenSignals devices, but no real device adapter or
  hardware validation exists.
- The simulator is physiologically plausible (OU jitter, RSA, scenarios), not a
  validated model of any person.
- Sensor dropout keeps the last BPM and marks the state invalid; the music keeps
  playing from the previous state rather than reacting to the dropout.

## Musical control

- MRT2 has **no tempo parameter**. BPM changes event scheduling, density,
  velocity and drum probability; perceived tempo follows the onset pattern. The
  model can still ignore or reinterpret that pattern, especially with high
  temperature or low CFG.
- The mapping is a simple monotonic design, not an aesthetic model. It was not
  tuned by listening tests or users.
- Onset rate and drum-active frames respond clearly (3.43 vs 0.63 onsets/s and
  2.23 vs 0.05 drum-active frames/s in the final demo),
  but "high BPM sounds faster/more energetic" is an indirect, model-dependent
  effect, not a guarantee.
- The legacy scheduler uses a fixed 16th-note grid and a fixed pitch-class
  profile from one MIDI transcription. The default `midi`/`hybrid` modes now
  follow the analysed 16-bar loop's chord changes, but the loop itself is fixed
  and repeats every 32 s; it does not follow the whole song form.
- High energy shortens note durations, so the mean number of *overlapping* notes
  can be lower at high BPM even though onsets are much more frequent. Event rate
  is the meaningful response metric.

## Iteration-2 musical layers

- The MIDI is a **Basic Pitch transcription**, not a score. It merges vocals and
  accompaniment into one track; there is no verified vocal/instrumental split.
  116 of 761 notes are non-diatonic and the upper voice is noisy. The extracted
  theme is a cleaned best effort, not a faithful melody.
- Harmonic analysis uses a diatonic-triad vocabulary. It will mislabel borrowed
  chords, inversions and short passing harmony. Low-confidence chords
  (~0.50) are treated as passing material.
- The selected loop (bars 44–59, Eb–Ab–Bb–Eb) is 32 s long and repeats. Longer
  sessions will sound cyclical unless a future version switches loops.
- The heartbeat is a fixed lub-dub/pulse pattern quantised to the live BPM. It
  is stable and exact, but it is not expressive: there is no swing, human
  timing, or physiological beat-to-beat variation beyond the smoothed BPM.
- The instrumental render uses a tiny additive numpy synth. It is clean and
  deterministic but deliberately simple; it is not a realistic piano/string
  sample library.
- MRT2 still colours the result and can still produce vocal-like or unstable
  artefacts. The redesign removes the lead melody and drum pulse from its
  conditioning and conditions it on an instrumental render, which reduces but
  does not eliminate the problem.
- The agent cannot subjectively hear audio. All claims here are numerical or
  structural; final musical acceptance requires a human listener.

## Reference song

- The recording is never mixed in, and the default conditioning source is now
  an **instrumental rendering of the MIDI transcription**, not the MP3. The MP3
  remains available for comparison and for the `--conditioning audio` ablation.
- Influence is still indirect:
  - 12-token MusicCoCa windows are coarse style descriptors.
  - The prefill seeds a 28 s excerpt of the 48 s MIDI render. A longer session
    drifts away from that context; there is no re-prefill or sliding audio
    context.
  - The loop repeats, so style windows derived from a longer render also repeat.
- The generated output is not expected to quote or reproduce the song's melody.
  Reference conditioning is a bias, not source separation or transcription.
- The pitch profile comes from `basic_pitch_transcription.mid`; its accuracy is
  not validated here (see `MIDI_ANALYSIS.md`).

## Model / runtime

- Only `mrt2_small` is usable. `mrt2_base` has a state file but no `.mlxfn` in the
  local assets.
- MLX is pinned to **0.31.1**; 0.32.x fails to import the shipped `.mlxfn`.
- The C++ Magenta runtime cannot be built on this machine (no Metal compiler /
  full Xcode). The Python/MLX path is the only verified path.
- Prefill requires a 650-frame forced pass: ~26 s one-time compute, 8.3 MB state
  cache, ~0.01 s cached load. It is not free at first startup.
- The SpectroStream encoder is traced for a fixed 28 s excerpt. Arbitrary-length
  audio prefill is not supported.

## Latency and playback

- Reported `latency_ms` is **inference time per 40 ms frame** (mean ~25 ms), not
  end-to-end acoustic latency.
- Playback adds the ring-buffer fill (~1.33 s in the acceptance run), a 20 ms
  block, and unmeasured CoreAudio/device latency. The practical delay from
  control change to speaker is therefore on the order of **1.3–1.5 s**, not
  25 ms. This is a deliberate stability trade-off, not a low-latency instrument.
- The heartbeat and accompaniment are rendered in the playback thread, so they
  continue through inference stalls; MRT2 itself will simply hold its last
  buffered material. A sustained underrun still affects only the MRT2 layer.
- On stop, the buffered tail is now drained and recorded
  (`drain_on_stop=True`), so the WAV includes the material that was queued for
  playback.
- The output path has a smoothed per-layer mix, a master gain, an
  instant-attack/smooth-release limiter and a hard clip guard. The 12 s real
  hybrid check had peak 0.654 and 0 clipped samples. This is a safety net, not
  an artistic mastering chain, and it does not guarantee perceived loudness or
  spectral balance.
- Underruns are faded rather than clicked, but a sustained underrun would still
  be audible as dropout. The acceptance run had zero underruns.
- Startup has a ~0.4 s pre-roll/device-start handoff gap in generation metrics;
  it is excluded from steady-state gap statistics. Audio is protected by the
  pre-roll.

## Reporting

- `report.json` stores one record per 40 ms frame for the whole session. A long
  session grows the JSON linearly (~25 records/s; ~90k records/hour). There is no
  downsampling or retention limit yet.
- The MIDI timeline can extend up to the 1 s scheduler lookahead past the audio
  end, so `timeline.duration_seconds` may be slightly longer than
  `session.audio_seconds`. This is scheduled-but-not-heard material, not a gap.
- The report's "high/low BPM" split is at the session median, not fixed clinical
  thresholds.
- End-to-end acoustic latency and subjective audio quality are not measured
  automatically. No human listening test was performed; validation is numerical
  (non-silence, peak/RMS, clipping) plus successful CoreAudio playback. Musical
  quality should be judged by a listener.
