"""Streaming MRT2 engine.

The exported ``.mlxfn`` is called one 40 ms frame at a time while carrying the
transformer state between calls.  That is what makes BPM changes continuous:
the model is never restarted, and a measurement does not create a new audio
clip.

Implemented here:
  * conditioning layout (12 style + 128 notes + 1 drum),
  * positive / negative CFG conditioning,
  * one-frame stepping with persistent state,
  * SpectroStream audio prefill (same procedure as the C++ runtime),
  * state save/load, warm-up, reset,
  * a deterministic mock engine for tests and headless pipeline checks.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from diastole._magenta import mx, paths
from diastole.config import (
    FRAME_SECONDS,
    NOTE_TOKEN_COUNT,
    SAMPLES_PER_FRAME,
    SAMPLE_RATE,
    STYLE_TOKEN_COUNT,
    TOKEN_OFFSET,
    EngineConfig,
)

Progress = Callable[[str], None] | None


def _as_conditioning(style, notes, drums) -> np.ndarray:
    style = list(style)
    notes = list(notes)
    drums = list(drums)
    if len(style) != STYLE_TOKEN_COUNT:
        raise ValueError(f"Expected {STYLE_TOKEN_COUNT} style tokens, got {len(style)}")
    if len(notes) != NOTE_TOKEN_COUNT:
        raise ValueError(f"Expected {NOTE_TOKEN_COUNT} note tokens, got {len(notes)}")
    if len(drums) != 1:
        raise ValueError(f"Expected 1 drum token, got {len(drums)}")
    return np.asarray(style + notes + drums, dtype=np.int32)


class MRT2Engine:
    """Thin, explicit wrapper around the exported MRT2 MLX function."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        state: list | None = None,
        warmup: bool = True,
    ) -> None:
        self.config = config or EngineConfig()
        self.model_size = self.config.model_size
        self.sample_rate = SAMPLE_RATE
        self.samples_per_frame = SAMPLES_PER_FRAME
        self.frame_seconds = FRAME_SECONDS
        self.last_step_ms = 0.0
        self.step_count = 0

        model_dir = paths.models_dir() / self.model_size
        mlxfn_path = model_dir / f"{self.model_size}.mlxfn"
        state_path = model_dir / f"{self.model_size}_state.safetensors"
        if not mlxfn_path.exists():
            raise FileNotFoundError(
                f"MRT2 model not found at {mlxfn_path}. "
                "Run scripts/link_assets.sh and check MAGENTA_HOME."
            )
        if not state_path.exists():
            raise FileNotFoundError(f"MRT2 initial state not found at {state_path}")

        t0 = time.time()
        self._fn = mx.import_function(str(mlxfn_path))
        data = mx.load(str(state_path))
        self._initial_state = [data[f"state_{i}"] for i in range(len(data))]
        mx.eval(self._initial_state)
        self.load_seconds = time.time() - t0
        self._previous_frame_index = self._find_previous_frame_index(self._initial_state)
        self._state = list(self._initial_state if state is None else state)

        if warmup and self.config.warmup_steps > 0:
            self.warmup(self.config.warmup_steps)

    # -- state ------------------------------------------------------------
    @staticmethod
    def _find_previous_frame_index(state: list) -> int:
        candidates = []
        for i, array in enumerate(state):
            shape = tuple(int(x) for x in array.shape)
            if shape == (1, 1, STYLE_TOKEN_COUNT) and array.dtype == mx.uint32:
                candidates.append(i)
        if candidates:
            return candidates[-1]
        for i, array in enumerate(state):
            if tuple(int(x) for x in array.shape) == (1, 1, STYLE_TOKEN_COUNT):
                return i
        raise RuntimeError("Could not locate the previous-frame state slot")

    @property
    def state(self) -> list:
        return self._state

    def set_state(self, state: list) -> None:
        if len(state) != len(self._initial_state):
            raise ValueError(
                f"Prefill state has {len(state)} arrays; model expects {len(self._initial_state)}"
            )
        self._state = list(state)

    def reset(self) -> None:
        self._state = list(self._initial_state)
        self.step_count = 0

    def load_state(self, path: str) -> list:
        data = mx.load(str(path))
        state = [data[f"state_{i}"] for i in range(len(data))]
        mx.eval(state)
        self.set_state(state)
        return state

    def save_state(self, path: str) -> None:
        mx.save_safetensors(
            str(path), {f"state_{i}": a for i, a in enumerate(self._state)}
        )

    # -- arguments --------------------------------------------------------
    def _build_args(self, style, notes, drums, forced: mx.array | None = None) -> list:
        raw = _as_conditioning(style, notes, drums)
        positive = (raw + TOKEN_OFFSET).reshape(1, 1, -1)
        style_start = 0
        style_end = STYLE_TOKEN_COUNT
        notes_start = STYLE_TOKEN_COUNT
        notes_end = STYLE_TOKEN_COUNT + NOTE_TOKEN_COUNT

        neg_musiccoca = raw.copy()
        neg_musiccoca[style_start:style_end] = -1
        neg_notes = raw.copy()
        neg_notes[notes_start:notes_end] = -1

        cfg = self.config
        if forced is None:
            forced_array = mx.zeros((1, 0, STYLE_TOKEN_COUNT), dtype=mx.int32)
        else:
            forced_array = forced

        return [
            mx.array(positive, dtype=mx.int32),
            mx.array([cfg.temperature], dtype=mx.float32),
            mx.array([cfg.top_k], dtype=mx.int32),
            mx.array([cfg.cfg_musiccoca], dtype=mx.float32),
            mx.array([cfg.cfg_notes], dtype=mx.float32),
            mx.array([cfg.cfg_drums], dtype=mx.float32),
            mx.array((neg_musiccoca + TOKEN_OFFSET).reshape(1, 1, -1), dtype=mx.int32),
            mx.array((neg_notes + TOKEN_OFFSET).reshape(1, 1, -1), dtype=mx.int32),
            forced_array,
        ]

    # -- generation -------------------------------------------------------
    def step(
        self,
        style_tokens,
        notes,
        drums,
        forced: mx.array | None = None,
        evaluate: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Generate one 40 ms frame and advance the streaming state."""

        args = self._build_args(style_tokens, notes, drums, forced)
        t0 = time.perf_counter()
        outputs = self._fn(args + self._state)
        if evaluate:
            mx.eval(outputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._state = list(outputs[1:])
        self.last_step_ms = elapsed_ms
        self.step_count += 1

        frame = np.asarray(outputs[0])
        samples = frame[0].T
        if np.issubdtype(samples.dtype, np.integer):
            samples = samples.astype(np.float32) / 32768.0
        else:
            samples = samples.astype(np.float32)
        if samples.shape != (self.samples_per_frame, 2):
            # Be forgiving about a (1, samples, channels) export variant.
            if samples.shape == (2, self.samples_per_frame):
                samples = samples.T
            else:
                samples = samples.reshape(-1, 2)
        return samples, elapsed_ms

    def warmup(self, steps: int = 2) -> None:
        style = [-1] * STYLE_TOKEN_COUNT
        notes = [-1] * NOTE_TOKEN_COUNT
        drums = [-1]
        state = list(self._state)
        t0 = time.perf_counter()
        for _ in range(max(0, int(steps))):
            args = self._build_args(style, notes, drums)
            outputs = self._fn(args + state)
            mx.eval(outputs)
            state = list(outputs[1:])
        self.warmup_seconds = time.perf_counter() - t0
        # Warm-up deliberately does not advance the real state.

    def prefill(
        self, tokens: np.ndarray, progress: Progress = None
    ) -> list:
        """Seed the transformer from SpectroStream token frames.

        ``tokens`` must have shape (frames, 12) with raw RVQ codes (0..1023).
        The unique-code offset ``k*1024 + 7`` used by the exported model is
        ``k*1024 + 6`` for forced prefill tokens (see the C++ runtime); the
        winning last frame is then injected into the previous-frame state slot.
        """

        tokens = np.asarray(tokens, dtype=np.int32)
        if tokens.ndim != 2 or tokens.shape[1] != STYLE_TOKEN_COUNT:
            raise ValueError(f"Expected prefill tokens (frames, 12), got {tokens.shape}")
        if tokens.shape[0] < 2:
            raise ValueError("Need at least two prefill frames")

        offsets = np.arange(STYLE_TOKEN_COUNT, dtype=np.int32) * 1024 + 6
        unique = (tokens + offsets[np.newaxis, :]).astype(np.int32)

        style = [-1] * STYLE_TOKEN_COUNT
        notes = [-1] * NOTE_TOKEN_COUNT
        drums = [-1]
        state = list(self._initial_state)
        total = unique.shape[0] - 1
        t0 = time.perf_counter()
        for i in range(total):
            forced = mx.array(unique[i].reshape(1, 1, -1), dtype=mx.int32)
            args = self._build_args(style, notes, drums, forced)
            outputs = self._fn(args + state)
            mx.eval(outputs)
            state = list(outputs[1:])
            if progress and ((i + 1) % 100 == 0 or i == total - 1):
                elapsed = time.perf_counter() - t0
                progress(
                    f"prefill {i + 1}/{total} frames "
                    f"({elapsed:.1f}s, {elapsed / (i + 1) * 1000:.1f} ms/frame)"
                )

        last = mx.array(unique[-1].reshape(1, 1, -1), dtype=mx.int32)
        idx = self._previous_frame_index
        slot = state[idx]
        if len(slot.shape) == 3:
            state[idx] = last.astype(slot.dtype)
        else:
            cfg_depth = int(slot.shape[1])
            state[idx] = mx.broadcast_to(
                last[:, None, :, :], (1, cfg_depth, 1, STYLE_TOKEN_COUNT)
            ).astype(slot.dtype)
        mx.eval(state)
        self.prefill_seconds = time.perf_counter() - t0
        return state

    def info(self) -> dict:
        return {
            "model_size": self.model_size,
            "sample_rate": self.sample_rate,
            "samples_per_frame": self.samples_per_frame,
            "frame_seconds": self.frame_seconds,
            "state_arrays": len(self._initial_state),
            "state_bytes": int(
                sum(int(np.prod(tuple(a.shape))) * a.dtype.size for a in self._initial_state)
            ),
            "load_seconds": round(self.load_seconds, 3),
            "warmup_seconds": round(getattr(self, "warmup_seconds", 0.0), 3),
            "cfg_scales": self.config.cfg_scales,
        }


class MockEngine:
    """Deterministic, model-free audio used by tests and ``--mock`` runs.

    The timbre is intentionally simple, but the mapping is real: note onsets
    choose the oscillator frequency, note density and drums change the
    amplitude, and phase is continuous across frames.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, seed: int = 7) -> None:
        self.sample_rate = int(sample_rate)
        self.samples_per_frame = int(round(FRAME_SECONDS * self.sample_rate))
        self.frame_seconds = FRAME_SECONDS
        self.model_size = "mock"
        self.last_step_ms = 0.0
        self.step_count = 0
        self._phase = 0.0
        self._phase2 = 0.0
        self._rng = np.random.default_rng(seed)
        self._initial_state: list = []
        self._state: list = []
        self.load_seconds = 0.0
        self.warmup_seconds = 0.0

    @property
    def state(self) -> list:
        return self._state

    def set_state(self, state: list) -> None:
        self._state = list(state)

    def reset(self) -> None:
        self._phase = 0.0
        self._phase2 = 0.0
        self.step_count = 0

    def load_state(self, path: str) -> list:
        return []

    def save_state(self, path: str) -> None:
        return None

    def prefill(self, tokens: np.ndarray, progress: Progress = None) -> list:
        if progress:
            progress("mock prefill (no-op)")
        return []

    def step(self, style_tokens, notes, drums, forced=None, evaluate: bool = True):
        t0 = time.perf_counter()
        notes = list(notes)
        active = [i for i, v in enumerate(notes) if v in (1, 2)]
        onsets = [i for i, v in enumerate(notes) if v == 2]
        pitch = onsets[0] if onsets else (active[0] if active else 60)
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        amp = 0.10 + 0.06 * min(6, len(active)) / 6.0
        if drums and int(drums[0]) == 1:
            amp += 0.05

        n = self.samples_per_frame
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        left = amp * np.sin(self._phase + math.tau * freq * t)
        right = 0.85 * amp * np.sin(self._phase2 + math.tau * freq * 0.997 * t + 0.4)
        self._phase = (self._phase + math.tau * freq * self.frame_seconds) % math.tau
        self._phase2 = (self._phase2 + math.tau * freq * 1.005 * self.frame_seconds) % math.tau
        samples = np.stack([left, right], axis=1).astype(np.float32)

        if drums and int(drums[0]) == 1:
            click = self._rng.normal(0, 0.03, size=(min(n, 480), 2)).astype(np.float32)
            samples[: click.shape[0]] += click
        samples = np.clip(samples, -1.0, 1.0)

        self.step_count += 1
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        return samples, self.last_step_ms

    def info(self) -> dict:
        return {
            "model_size": "mock",
            "sample_rate": self.sample_rate,
            "samples_per_frame": self.samples_per_frame,
            "frame_seconds": self.frame_seconds,
            "state_arrays": 0,
            "cfg_scales": {},
        }
