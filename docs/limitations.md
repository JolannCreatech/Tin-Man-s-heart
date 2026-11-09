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
- The scheduler uses a fixed 16th-note grid and a fixed pitch-class profile from
  one MIDI transcription. It does not follow the song's actual chord changes.
- High energy shortens note durations, so the mean number of *overlapping* notes
  can be lower at high BPM even though onsets are much more frequent. Event rate
  is the meaningful response metric.

## Reference song

- The recording is never mixed in, but its influence is indirect:
  - 12-token MusicCoCa windows are coarse style descriptors.
  - The prefill seeds a 28 s excerpt (15–43 s of the song). A longer session
    drifts away from that context; there is no re-prefill or sliding audio
    context.
  - The default session walks the song at 1× from offset 0, so a 42 s run uses
    only the first ~42 s of the 170.5 s recording. The full set of 35 windows is
    cached for longer runs.
- The generated output is not expected to quote or reproduce the song's melody.
  Reference conditioning is a bias, not source separation or transcription.
- The pitch profile comes from a prior `basic_pitch_transcription.mid`; its
  accuracy is not validated here.

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
- The buffered tail is not drained on stop (`drain_on_stop` is reserved but
  unused). In the 42 s run, 1,954,560 frames were played from 2,017,920 recorded
  frames: about 1.3 s remains in the ring/WAV at shutdown and is not heard.
- There is no output limiter or normalization. Observed peaks ranged from 0.78
  to 0.98 across runs (no clipped samples, but close to full scale); other
  settings could clip. The WAV is written as PCM16.
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
