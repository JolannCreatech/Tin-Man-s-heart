"""Central configuration and constants for DIASTOLE.

Everything that has a physical meaning (frame rate, BPM limits, buffer sizes)
lives here so the mapping and the report can quote exactly what was used.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository / asset locations
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = REPO_ROOT / "assets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache"
REFERENCE_AUDIO = REPO_ROOT / "music" / "The Wizard Of Oz - If I Only Had A Heart.mp3"
REFERENCE_MIDI = REPO_ROOT / "music" / "basic_pitch_transcription.mid"

# ---------------------------------------------------------------------------
# MRT2 constants (verified from the Magenta RealTime 2 source)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48_000
FRAME_RATE = 25
SAMPLES_PER_FRAME = 1_920  # 40 ms at 48 kHz
FRAME_SECONDS = SAMPLES_PER_FRAME / SAMPLE_RATE  # 0.04

STYLE_TOKEN_COUNT = 12
NOTE_TOKEN_COUNT = 128
DRUM_TOKEN_COUNT = 1
CONDITIONING_WIDTH = STYLE_TOKEN_COUNT + NOTE_TOKEN_COUNT + DRUM_TOKEN_COUNT

# The exported .mlxfn adds NUM_RESERVED_TOKENS + 1 (dropout) to raw tokens.
TOKEN_OFFSET = 7
MASK_RAW = -1

NUM_MIDI_PITCHES = 128
DEFAULT_PPQ = 480


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    """Parameters for the local MRT2 engine."""

    model_size: str = "mrt2_small"
    temperature: float = 1.1
    top_k: int = 50
    cfg_musiccoca: float = 3.0
    cfg_notes: float = 2.0
    cfg_drums: float = 1.0
    warmup_steps: int = 2
    prefill: bool = True
    prefill_excerpt_start: float = 15.0
    prefill_excerpt_seconds: float = 28.0

    @property
    def cfg_scales(self) -> dict[str, float]:
        return {
            "musiccoca": self.cfg_musiccoca,
            "notes": self.cfg_notes,
            "drums": self.cfg_drums,
        }

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ReferenceConfig:
    """How the source recording is ingested."""

    audio_path: Path = REFERENCE_AUDIO
    midi_path: Path = REFERENCE_MIDI
    cache_dir: Path = DEFAULT_CACHE_DIR
    style_clip_seconds: float = 10.0
    style_hop_seconds: float = 5.0
    style_speed: float = 1.0  # how fast we walk through the song
    style_offset_seconds: float = 0.0
    prefill_excerpt_start: float = 15.0
    prefill_excerpt_seconds: float = 28.0
    prefill_normalize_peak: float = 0.9
    pitch_register_low: int = 48
    pitch_register_high: int = 76

    # --- iteration-2 conditioning options -----------------------------
    # "audio"    -> the file at ``audio_path`` (default: the MP3 recording)
    # "midi"     -> an instrumental rendering of the MIDI, generated on demand
    conditioning_source: str = "midi"
    style_mode: str = "fixed"  # drift | fixed | loop
    style_position_seconds: float = 0.0
    style_speed: float = 1.0
    text_style: str = "solo piano and warm strings, instrumental, no vocals"
    text_style_blend: float = 0.65  # 0 = audio only, 1 = text only
    text_style_seed: int = 0
    midi_render_seconds: float = 48.0
    midi_render_cache: Path = DEFAULT_CACHE_DIR / "reference"

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["audio_path"] = str(self.audio_path)
        d["midi_path"] = str(self.midi_path)
        d["cache_dir"] = str(self.cache_dir)
        d["midi_render_cache"] = str(self.midi_render_cache)
        return d


@dataclasses.dataclass(frozen=True)
class ProcessorConfig:
    """Heart-rate signal conditioning."""

    min_bpm: float = 30.0
    max_bpm: float = 220.0
    max_sample_age: float = 3.0
    median_window: int = 5
    ema_tau: float = 1.8
    max_slew_bpm_per_s: float = 12.0
    variability_window: int = 12
    arousal_low: float = 50.0
    arousal_high: float = 150.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MappingConfig:
    """BPM -> musical parameters. All monotonic, all bounded."""

    tempo_base: float = 30.0
    tempo_gain: float = 0.65
    tempo_min: float = 50.0
    tempo_max: float = 180.0
    density_base: float = 0.20
    density_gain: float = 0.65
    note_activity_base: float = 0.25
    note_activity_gain: float = 0.60
    dynamics_base: float = 0.45
    dynamics_gain: float = 0.50
    intensity_base: float = 0.15
    intensity_gain: float = 0.75
    drum_gate_threshold: float = 0.28
    subdivision_low: int = 2
    subdivision_high: int = 4

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SchedulerConfig:
    """Rolling 16th-note lookahead scheduler."""

    lookahead_seconds: float = 1.0
    seed: int = 7
    steps_per_beat: int = 4
    steps_per_bar: int = 16

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ArrangementConfig:
    """BPM-locked arrangement of the MIDI-derived musical material.

    ``mode`` selects the behaviour:
      * ``legacy``  - the original probabilistic timeline (iteration 1);
      * ``midi``    - deterministic MIDI-derived bass/pad/melody, no heartbeat;
      * ``hybrid``  - MIDI-derived material plus the deterministic heartbeat.
    """

    mode: str = "hybrid"
    heartbeat: bool = True
    heartbeat_pattern: str = "heartbeat"  # heartbeat | pulse | kick
    heartbeat_subdivision: int = 2  # ticks per beat; 0 disables ticks
    accent_every: int = 4  # beats
    heartbeat_volume_accent: float = 0.22
    beats_per_bar: int = 4
    harmonic_rhythm_beats: int = 2
    enable_bass: bool = True
    enable_pad: bool = True
    enable_melody: bool = True
    melody_density: float = 0.60
    pad_density: float = 0.90
    bass_octave: int = 2
    loop_start_bar: int | None = None
    loop_bars: int = 16
    theme_bars: int = 16
    quantize_beats: float = 0.25
    humanise_seconds: float = 0.006
    seed: int = 7

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MixConfig:
    """Per-layer gains, ducking and limiting for the output mixer."""

    heartbeat_gain: float = 0.62
    accompaniment_gain: float = 0.34
    mrt2_gain: float = 0.52
    reference_gain: float = 0.0  # disabled by default
    master_gain: float = 0.92
    limiter_threshold: float = 0.94
    limiter_release_seconds: float = 0.25
    ducking: float = 0.0  # 0..1; sidechain MRT2/accompaniment from heartbeat
    duck_attack_seconds: float = 0.008
    duck_release_seconds: float = 0.22
    smooth_gain_seconds: float = 0.20
    record_stems: bool = False
    reference_audio: Path | None = None

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["reference_audio"] = str(self.reference_audio) if self.reference_audio else None
        return d


@dataclasses.dataclass(frozen=True)
class AudioConfig:
    """Playback and recording."""

    enabled: bool = True
    device: int | str | None = None
    sample_rate: int = SAMPLE_RATE
    channels: int = 2
    block_seconds: float = 0.02
    buffer_seconds: float = 1.6
    start_threshold_seconds: float = 0.75
    target_fill_fraction: float = 0.85
    drain_on_stop: bool = True

    @property
    def block_frames(self) -> int:
        return max(1, round(self.block_seconds * self.sample_rate))

    @property
    def buffer_frames(self) -> int:
        return round(self.buffer_seconds * self.sample_rate)

    @property
    def start_threshold_frames(self) -> int:
        return round(self.start_threshold_seconds * self.sample_rate)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["enabled"] = self.enabled
        return d


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """Top-level configuration for one session."""

    engine: EngineConfig = dataclasses.field(default_factory=EngineConfig)
    reference: ReferenceConfig = dataclasses.field(default_factory=ReferenceConfig)
    processor: ProcessorConfig = dataclasses.field(default_factory=ProcessorConfig)
    mapping: MappingConfig = dataclasses.field(default_factory=MappingConfig)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    arrangement: ArrangementConfig = dataclasses.field(default_factory=ArrangementConfig)
    mix: MixConfig = dataclasses.field(default_factory=MixConfig)
    audio: AudioConfig = dataclasses.field(default_factory=AudioConfig)
    scenario: str = "rest"
    duration: float | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int | None = None

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["engine"] = self.engine.as_dict()
        d["reference"] = self.reference.as_dict()
        d["processor"] = self.processor.as_dict()
        d["mapping"] = self.mapping.as_dict()
        d["scheduler"] = self.scheduler.as_dict()
        d["arrangement"] = self.arrangement.as_dict()
        d["mix"] = self.mix.as_dict()
        d["audio"] = self.audio.as_dict()
        d["output_dir"] = str(self.output_dir)
        return d


def ensure_magenta_home() -> Path:
    """Point MAGENTA_HOME at this repository's symlinked assets by default.

    ``magenta_rt.paths`` reads the environment variable at import time, so this
    must run before importing magenta_rt.  The generation module does that.
    """

    current = os.environ.get("MAGENTA_HOME")
    if current:
        return Path(current)
    os.environ["MAGENTA_HOME"] = str(DEFAULT_ASSETS)
    return DEFAULT_ASSETS


def cache_path(name: str) -> Path:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR / name
