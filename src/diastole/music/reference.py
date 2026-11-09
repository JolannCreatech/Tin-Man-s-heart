"""Reference-song conditioning.

The recording of *If I Only Had a Heart* influences generation in two genuine
ways:

1. MusicCoCa embeds 10 s windows of the recording and the resulting 12 RVQ
   style tokens are fed to MRT2 as it plays.  The window advances with session
   time, so the harmonic/timbral character drifts through the song.
2. A 28 s excerpt is encoded by SpectroStream into RVQ tokens and prefilled
   into the transformer state before generation starts.  This is the same
   mechanism the C++ runtime calls ``prefill_state``.

The raw recording is never mixed into the output.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from diastole import config as cfg
from diastole.config import ReferenceConfig
from diastole.music.pitch import PitchProfile


def _file_identity(path: Path) -> str:
    try:
        st = path.stat()
        return f"{path}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return str(path)


def _digest(*parts: object) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(samples, dtype=np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // g
    down = int(sr_in) // g
    # resample_poly works along axis 0 for (samples, channels).
    out = resample_poly(samples, up, down, axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def load_audio(path: str | Path, sample_rate: int = cfg.SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Read an audio file as float32 stereo at ``sample_rate``."""

    import soundfile as sf

    path = Path(path)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.concatenate([data, data], axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    data = _resample(data, int(sr), int(sample_rate))
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    return data, int(sample_rate)


class ReferenceMusic:
    """Loader, style-window cache, prefill cache and pitch profile."""

    def __init__(self, config: ReferenceConfig | None = None) -> None:
        self.config = config or ReferenceConfig()
        self._audio: np.ndarray | None = None
        self._audio_sr: int = cfg.SAMPLE_RATE
        self._style_tokens: np.ndarray | None = None
        self._style_starts: np.ndarray | None = None
        self._pitch_profile: PitchProfile | None = None
        self._prefill_tokens: np.ndarray | None = None
        self._prefill_meta: dict = {}
        self._prefill_state_path: Path | None = None
        self.prefill_cache_hit: bool | None = None
        self.prefill_load_seconds = 0.0
        self.prefill_compute_seconds = 0.0
        self._identity = _file_identity(Path(self.config.audio_path))
        self._style_digest = _digest(
            self._identity,
            self.config.style_clip_seconds,
            self.config.style_hop_seconds,
        )
        self._prefill_digest = _digest(
            self._identity,
            self.config.prefill_excerpt_start,
            self.config.prefill_excerpt_seconds,
            self.config.prefill_normalize_peak,
        )
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    # -- audio ------------------------------------------------------------
    @property
    def audio(self) -> np.ndarray:
        if self._audio is None:
            self.load_audio()
        assert self._audio is not None
        return self._audio

    @property
    def sample_rate(self) -> int:
        return self._audio_sr

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def load_audio(self) -> np.ndarray:
        self._audio, self._audio_sr = load_audio(self.config.audio_path)
        return self._audio

    # -- pitch profile ----------------------------------------------------
    def pitch_profile(self, force: bool = False) -> PitchProfile:
        if self._pitch_profile is None or force:
            self._pitch_profile = PitchProfile.from_midi(
                self.config.midi_path,
                register_low=self.config.pitch_register_low,
                register_high=self.config.pitch_register_high,
            )
        return self._pitch_profile

    # -- style tokens -----------------------------------------------------
    @property
    def style_cache_path(self) -> Path:
        return Path(self.config.cache_dir) / f"reference_style_{self._style_digest}.npz"

    @property
    def style_tokens(self) -> np.ndarray:
        if self._style_tokens is None:
            self.prepare_style_tokens()
        assert self._style_tokens is not None
        return self._style_tokens

    def prepare_style_tokens(
        self, force: bool = False, progress: callable | None = None
    ) -> np.ndarray:
        path = self.style_cache_path
        if path.exists() and not force:
            data = np.load(path)
            self._style_tokens = data["tokens"].astype(np.int32)
            self._style_starts = data["starts"].astype(np.float64)
            return self._style_tokens

        if progress:
            progress("Embedding reference audio with MusicCoCa (first run only)...")
        return self._embed_style_tokens(progress=progress)

    def _embed_style_tokens(self, progress: callable | None = None) -> np.ndarray:
        from magenta_rt import audio as maudio
        from magenta_rt.musiccoca import MusicCoCa

        if progress:
            progress("Loading MusicCoCa...")
        model = MusicCoCa()
        wave = maudio.Waveform(self.audio, self.sample_rate)
        if progress:
            progress("Embedding reference windows...")
        embeddings = model.embed(
            wave,
            pool_across_time=False,
            hop_length=float(self.config.style_hop_seconds),
        )
        embeddings = np.asarray(embeddings)
        if embeddings.ndim == 2:
            embeddings = embeddings[np.newaxis, ...]
        tokens = np.asarray(model.tokenize(embeddings[0]), dtype=np.int32)
        starts = np.arange(tokens.shape[0], dtype=np.float64) * float(
            self.config.style_hop_seconds
        )
        np.savez(
            self.style_cache_path,
            tokens=tokens,
            starts=starts,
            clip_seconds=self.config.style_clip_seconds,
            hop_seconds=self.config.style_hop_seconds,
        )
        self._style_tokens = tokens
        self._style_starts = starts
        return tokens

    def style_index_at(self, session_time: float) -> int:
        tokens = self.style_tokens
        if tokens.shape[0] == 0:
            raise RuntimeError("Reference style token cache is empty")
        song_time = (
            float(self.config.style_offset_seconds)
            + float(self.config.style_speed) * max(0.0, float(session_time))
        ) % max(1e-6, self.duration)
        index = int(song_time // max(1e-6, self.config.style_hop_seconds))
        return min(tokens.shape[0] - 1, max(0, index))

    def style_tokens_at(self, session_time: float) -> list[int]:
        index = self.style_index_at(session_time)
        return [int(v) for v in self.style_tokens[index]]

    def style_window_at(self, session_time: float) -> dict:
        index = self.style_index_at(session_time)
        starts = self._style_starts
        start = float(starts[index]) if starts is not None else index * self.config.style_hop_seconds
        return {
            "index": index,
            "start_seconds": round(start, 2),
            "end_seconds": round(start + self.config.style_clip_seconds, 2),
            "tokens": self.style_tokens_at(session_time),
        }

    def style_tokens_for_session(self, session_time: float) -> list[int]:
        """Deterministic mask when no style cache has been prepared.

        This deliberately never triggers MusicCoCa work from the generation
        thread; ``prepare_style_tokens`` is called once during startup.
        """

        if self._style_tokens is None:
            return [-1] * cfg.STYLE_TOKEN_COUNT
        try:
            return self.style_tokens_at(session_time)
        except Exception:
            return [-1] * cfg.STYLE_TOKEN_COUNT

    # -- prefill ----------------------------------------------------------
    @property
    def prefill_tokens_cache_path(self) -> Path:
        return Path(self.config.cache_dir) / f"reference_prefill_tokens_{self._prefill_digest}.npy"

    @property
    def prefill_state_cache_path(self) -> Path:
        if self._prefill_state_path is not None:
            return self._prefill_state_path
        return (
            Path(self.config.cache_dir)
            / f"prefill_state_{self._prefill_digest}.safetensors"
        )

    def prefill_tokens(self, force: bool = False, progress: callable | None = None) -> np.ndarray:
        if self._prefill_tokens is not None and not force:
            return self._prefill_tokens
        path = self.prefill_tokens_cache_path
        if path.exists() and not force:
            self._prefill_tokens = np.load(path).astype(np.int32)
            return self._prefill_tokens

        from diastole._magenta import mx, paths

        if progress:
            progress("Encoding the reference excerpt with SpectroStream...")
        encoder_path = paths.spectrostream_dir() / "spectrostream_encoder.mlxfn"
        encoder = mx.import_function(str(encoder_path))

        audio = self.audio
        sr = self.sample_rate
        start = int(max(0, round(self.config.prefill_excerpt_start * sr)))
        length = int(round(self.config.prefill_excerpt_seconds * sr))
        clip = audio[start : start + length]
        if clip.shape[0] < length:
            pad = np.zeros((length - clip.shape[0], clip.shape[1]), dtype=np.float32)
            clip = np.concatenate([clip, pad], axis=0)
        clip = np.asarray(clip, dtype=np.float32)
        peak = float(np.max(np.abs(clip))) if clip.size else 0.0
        if peak > 0:
            clip = clip * (float(self.config.prefill_normalize_peak) / peak)

        t0 = time.time()
        encoded = np.asarray(encoder(mx.array(clip[np.newaxis, ...], dtype=mx.float32)))
        tokens = encoded[0, 0, :, : cfg.STYLE_TOKEN_COUNT].astype(np.int32)
        trim_front = 25
        trim_back = 25
        tokens = tokens[trim_front : tokens.shape[0] - trim_back]
        if progress:
            progress(
                f"SpectroStream: {tokens.shape[0]} frames in {time.time() - t0:.1f}s"
            )
        np.save(path, tokens)
        self._prefill_tokens = tokens
        self._prefill_meta = {
            "frames": int(tokens.shape[0]),
            "encode_seconds": round(time.time() - t0, 2),
            "start_seconds": self.config.prefill_excerpt_start,
            "excerpt_seconds": self.config.prefill_excerpt_seconds,
        }
        return tokens

    def prefill_state(
        self,
        engine: object,
        force: bool = False,
        progress: callable | None = None,
    ) -> list:
        """Return (and cache) the transformer state produced by prefill."""

        from diastole._magenta import mx

        model = getattr(engine, "model_size", getattr(engine, "name", "unknown"))
        tokens = self.prefill_tokens(force=force, progress=progress)
        tokens_digest = _digest(int(tokens.shape[0]), int(tokens.sum()), self._prefill_digest)
        path = Path(self.config.cache_dir) / f"prefill_state_{model}_{tokens_digest}.safetensors"

        if path.exists() and not force:
            if progress:
                progress(f"Loading cached prefill state: {path.name}")
            t_load = time.perf_counter()
            data = mx.load(str(path))
            state = [data[f"state_{i}"] for i in range(len(data))]
            mx.eval(state)
            self._prefill_state_path = path
            self.prefill_cache_hit = True
            self.prefill_load_seconds = time.perf_counter() - t_load
            self.prefill_compute_seconds = 0.0
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    self.prefill_compute_seconds = float(
                        meta.get("compute_seconds", 0.0)
                    )
                except Exception:
                    pass
            return state

        if not hasattr(engine, "prefill"):
            raise TypeError("Engine does not implement prefill()")
        if progress:
            progress(f"Prefilling {tokens.shape[0]} frames (one-time, ~30 s)...")
        state = engine.prefill(tokens, progress=progress)
        mx.eval(state)
        mx.save_safetensors(str(path), {f"state_{i}": a for i, a in enumerate(state)})
        self._prefill_state_path = path
        self.prefill_cache_hit = False
        self.prefill_load_seconds = 0.0
        self.prefill_compute_seconds = float(getattr(engine, "prefill_seconds", 0.0))
        meta = {
            "model": model,
            "frames": int(tokens.shape[0]),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": str(self.config.audio_path),
            "excerpt_start_seconds": self.config.prefill_excerpt_start,
            "excerpt_seconds": self.config.prefill_excerpt_seconds,
            "state_arrays": len(state),
            "compute_seconds": round(
                float(getattr(engine, "prefill_seconds", 0.0)), 2
            ),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return state

    # -- reporting --------------------------------------------------------
    def info(self) -> dict:
        info = {
            "audio_path": str(self.config.audio_path),
            "audio_exists": Path(self.config.audio_path).exists(),
            "midi_path": str(self.config.midi_path),
            "midi_exists": Path(self.config.midi_path).exists(),
            "cache_dir": str(self.config.cache_dir),
            "style_clip_seconds": self.config.style_clip_seconds,
            "style_hop_seconds": self.config.style_hop_seconds,
        }
        if self._audio is not None:
            info["duration_seconds"] = round(self.duration, 3)
            info["sample_rate"] = self.sample_rate
        if self._style_tokens is not None:
            info["style_windows"] = int(self._style_tokens.shape[0])
        if self._prefill_tokens is not None:
            info["prefill_frames"] = int(self._prefill_tokens.shape[0])
        info["pitch_profile"] = self.pitch_profile().as_dict()
        return info
