# MIDI analysis and musical material

DIASTOLE now treats `music/basic_pitch_transcription.mid` as the primary
musical reference. The MP3 is kept for comparison and for the `--conditioning
audio` ablation only. This document records what the MIDI actually contains,
what was extracted, and how each extracted element is used.

## What the file is (and is not)

`basic_pitch_transcription.mid` is a **Basic Pitch transcription**, not a
score or a multi-stem production file.

| Property | Value |
|---|---|
| Format | SMF format 1, 480 PPQ |
| Tracks | 2 (track 0 has no notes; track 1 has all 761 notes) |
| Program | 4 (electric piano) on channel 0 |
| Tempo | 120.0 BPM, constant |
| Time signature | 4/4 |
| Duration | 169.36 s / 338.7 beats (84.7 bars) |
| Notes | 761 after parsing, 0 dropped |
| Separate vocal track | **No** |

Basic Pitch merges vocals and accompaniment into one track. There is therefore
no honest "vocal stem" to remove. The analysis instead derives a best-effort
upper voice (theme) and a bass line, and treats the upper voice as *material to
be cleaned*, not as a verified melody.

### Transcription issues found

- 116 of 761 notes are non-diatonic to the estimated key. Many are octave
  errors, passing tones, or short false positives in the upper register.
- Median note duration is **0.197 s**, mean **0.230 s** — very short, typical of
  a piano/vocal transcription rather than sustained parts.
- 16 notes have low velocity; 7 are above the useful register.
- No note was so short that it had to be dropped, but the upper line is noisy
  enough that using every note directly would produce clutter.

The analysis writes all diagnostics to `midi_analysis.json` in each session
directory (`issues`, `stats`, `bars`, `chords`, `theme`, `loop`).

## Analysis pipeline

Implemented in `src/diastole/music/midi_analysis.py`:

1. **Parse** the SMF with `mido`, build a tempo map, and convert ticks to beats
   and seconds. A numeric inversion of the tempo map is used (the initial
   implementation collapsed the song to 4 beats because it only knew about
   tempo events at tick 0; fixed by passing the MIDI end tick).
2. **Estimate key** with Krumhansl–Schmuckler pitch-class correlation over
   duration-weighted notes.
3. **Infer chords** on a diatonic triad vocabulary constrained to the estimated
   key, with a bass-anchored Viterbi pass. Each chord carries a confidence
   derived from the emission margin.
4. **Segment bars** from the tempo/time-signature grid, recording chord, bass,
   upper voice, note count and density per bar.
5. **Select a 16-bar loop** by scoring candidate windows for chord coverage,
   harmonic change, note density and thematic continuity.
6. **Extract a cleaned theme** from the loop: one upper-voice note per
   quantised cell, repeated identical pitches merged, low-confidence noise
   suppressed.

The selection is deterministic for a given file and `loop_bars`.

## Findings

### Key and pitch content

- Estimated key: **Eb major**, Krumhansl correlation **0.918** at the
  pitch-class level (the reported scalar confidence is 0.791 after the margin
  term).
- Duration-weighted pitch classes: **Eb 21.4%**, **Ab 17.0%**, **Bb 15.4%**,
  C 10.2%, G 8.9%, F 7.1%. The tonic triad and the IV/V degrees dominate, as
  expected for this song.

### Texture

| Metric | Value |
|---|---|
| Note count | 761 |
| Onsets per second | 4.49 |
| Mean polyphony | 1.03 |
| Maximum simultaneous notes | 6 |
| Pitch range | 34–98 (Bb1–D7) |
| Median pitch | 56 |

The low mean polyphony confirms that this is mostly a single melodic line with
occasional accompaniment, not a dense arrangement.

### Harmony

- **71 chord segments**, 0.82 chord changes per bar.
- The loop contains Eb–Ab–Bb–Eb material with several short passing chords.
- Chord confidence separates cleanly: the stable tonic/IV regions score
  0.87–0.94, while two-beat passing segments sit at ~0.50. Those low-confidence
  segments are treated as passing harmony, not as new tonal centres.

### Selected loop

The automatic 16-bar loop is **bars 44–59** (beats 176–240, 88.0–120.0 s at
120 BPM):

```
bar 44  Eb      bar 48  Ab      bar 52  Ab      bar 56  Ab
bar 45  Eb      bar 49  Ab      bar 53  Bb      bar 57  Bb
bar 46  Ab      bar 50  Bb      bar 54  Eb      bar 58  Eb
bar 47  Ab      bar 51  Eb      bar 55  Ab      bar 59  Eb
```

- 64 beats / 32 s long.
- 191 transcribed notes fall inside it; 53 survive theme cleaning.
- The progression is strongly tonal and returns to Eb, which makes it safe to
  loop under a physiological tempo clock.

## How the material is used

`src/diastole/music/midi_render.py` turns the loop into an instrumental
arrangement with a small deterministic numpy synth:

- **Bass** — chord bass pitch (with an optional fifth on alternating harmonic
  pulses), short attack, low register.
- **Pad** — three-note diatonic voicing around C4 built from the inferred
  chord, slow attack/release.
- **Lead** — cleaned theme notes, plucked envelope.

The same material is emitted live by `src/diastole/music/arranger.py` against
the BPM clock, so the accompaniment follows the heart rate even though the
underlying loop is fixed. The heartbeat is deliberately *not* part of the MIDI
material; it is a separate, BPM-exact layer.

The rendered loop is also the default conditioning source for MRT2:

- MusicCoCa embeds 10 s windows of the instrumental render (10 windows for a
  48 s render) and blends them with a MusicCoCa text embedding.
- SpectroStream encodes a 28 s excerpt of the render for transformer prefill.

This removes the lead vocal and the noisy MP3 mix from the conditioning signal
while preserving the song's key, harmony and melodic contour.
