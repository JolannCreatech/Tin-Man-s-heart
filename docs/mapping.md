# Physiological → musical mapping

The mapping is deterministic, bounded and monotonic. All limits live in
`src/diastole/config.py` so the report can quote exactly what was used.

## 1. Heart-rate processing

`BpmProcessor.update()` runs on every generation frame (25 Hz), but only a new
sensor timestamp moves the estimate. Re-polling a frozen reading cannot dominate
the filter.

```
raw sample
  -> reject if invalid / non-finite / outside 30–220 BPM / older than 3 s
  -> ignore duplicate timestamps
  -> median over the last 5 accepted readings
  -> EMA with tau = 1.8 s
  -> slew limit on the EMA output: 12 BPM/s
  -> PhysiologicalState
```

- `arousal = smoothstep(50, 150, smoothed_bpm)`, clamped to 0..1.
- `trend_bpm_per_s` is a smoothed derivative; `variability` is the standard
  deviation of recent raw readings.
- Invalid samples keep the last good BPM but mark the state `valid=False`, so
  the dashboard can show a disconnected sensor without tearing the music.

The slew limit is applied to the filter **output**. Limiting the filter input
instead would make a step response slower than the configured limit because the
EMA would keep re-scaling the capped delta. The output clamp gives an expressive
but bounded rise: the demo scenario moves from ~62 to ~133 BPM over its 6 s
ramp, then recovers.

## 2. Musical parameters

| Parameter | Formula | At 60 BPM | At 130 BPM |
|---|---|---|---|
| tempo | `30 + 0.65·bpm`, clamped 50–180 | 69 | 114.5 |
| energy | `arousal` | ~0.04 | ~0.90 |
| rhythmic density | `0.20 + 0.65·arousal` | 0.23 | 0.79 |
| note activity | `0.25 + 0.60·arousal` | 0.27 | 0.79 |
| dynamics | `0.45 + 0.50·arousal` | 0.47 | 0.90 |
| rhythmic intensity | `0.15 + 0.75·arousal` | 0.18 | 0.83 |
| drum gate | `intensity >= 0.28` | off | on |
| subdivision | 2 below arousal 0.5, else 4 | 2 | 4 |

This is the initial mapping, not a claim about musical taste. It is intentionally
simple and inspectable, and every value is recorded per frame in `report.json`.

## 3. Scheduler and timeline

`TimelineScheduler` appends events on a 16th-note grid using the **latest**
parameters for a 1 s lookahead. It never rewrites past events.

- Beat period comes from the mapped tempo. Notes are quantized to 16th steps.
- Position weights: downbeat 1.0, other beats 0.85, even 16ths 0.55, odd 16ths
  0.30. At subdivision 2, odd 16ths are skipped.
- At each bar a new three-note chord is chosen from the reference pitch-class
  profile; the downbeat voices the chord, other steps choose a melody pitch from
  the same profile.
- Note duration depends on energy (shorter at high energy); velocity depends on
  dynamics.
- Drums: kick on steps 0/8, snare on 4/12, hats on even steps, occasional open
  hat on 14. Each hit is gated by `rhythmic_intensity` and `drum_gate`.
- If inference stalls, the scheduler advances past missed steps instead of
  dumping a burst of late notes.

Because tempo is an event-scheduling property, perceived tempo follows the
onset pattern even though MRT2's clock is fixed at 25 Hz.

## 4. MIDI and 25 Hz conditioning

`MusicalTimeline` writes a type-1 SMF with a real tempo map. The file is
atomically replaced every 2 audio seconds, so one session always has one growing
MIDI artifact rather than one file per BPM sample.

MRT2 cannot read MIDI. `FrameAdapter` samples the same timeline for the frame's
40 ms window:

| Token | Semantics |
|---|---|
| note `-1` | masked (model decides) |
| note `0` | explicit off |
| note `1` | sustaining |
| note `2` | onset |
| note `3` | free |
| drum `-1` | masked |
| drum `0` | no hit |
| drum `1` | hit |

The pipeline uses explicit `0` for unused notes (not masked), so heart-rate
control actually drives the note pattern instead of handing silence back to the
model. Active pitches become `1`, pitches with an onset in the next 40 ms become
`2`.

The MusicCoCa style tokens are separate: 12 tokens from the reference window
active at the current session time, or `-1` for all 12 if reference preparation
failed.

## 5. What the report measures

The per-frame series records raw and smoothed BPM, trend, all musical
parameters, active notes, onsets, drum hits, inference time, buffer fill,
underruns and the active style-window index. The summary splits frames at the
median BPM and reports:

- onset rate (`onsets / 0.04 s`) and drum-active frame rate (frames containing at
  least one hit, per second), overall and per segment;
- MIDI note/drum events per second;
- density and active-note means per segment;
- the timeline's tempo changes, event counts and pitch-class histogram.

Event rate is the clearest response metric. Active-note count can be lower at
high BPM because high energy shortens note durations, so fewer notes overlap even
though far more onsets occur.
