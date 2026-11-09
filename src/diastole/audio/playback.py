"""Real-time audio path: SPSC ring buffer, playback thread and WAV recorder."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from diastole.config import AudioConfig


class RingBuffer:
    """Single-producer / single-consumer float32 ring buffer.

    ``write`` is called only by the generation thread and ``read`` only by the
    playback thread.  A small lock keeps the indices consistent; contention is
    negligible at 25 Hz.
    """

    def __init__(self, capacity_frames: int, channels: int = 2) -> None:
        self.capacity = int(max(1, capacity_frames))
        self.channels = int(channels)
        self._data = np.zeros((self.capacity, self.channels), dtype=np.float32)
        self._read = 0
        self._write = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        with self._lock:
            return self._size

    @property
    def free(self) -> int:
        with self._lock:
            return self.capacity - self._size

    @property
    def fill_fraction(self) -> float:
        with self._lock:
            return self._size / self.capacity

    def clear(self) -> None:
        with self._lock:
            self._read = self._write = self._size = 0
            self._data.fill(0.0)

    def write(self, samples: np.ndarray) -> int:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        if samples.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {samples.shape[1]}")
        with self._lock:
            n = min(samples.shape[0], self.capacity - self._size)
            if n <= 0:
                return 0
            first = min(n, self.capacity - self._write)
            self._data[self._write : self._write + first] = samples[:first]
            remaining = n - first
            if remaining > 0:
                self._data[:remaining] = samples[first:n]
            self._write = (self._write + n) % self.capacity
            self._size += n
            return n

    def read(self, frames: int, fill: bool = True) -> tuple[np.ndarray, bool]:
        """Return ``frames`` samples; the second value is True on underrun."""

        frames = int(max(0, frames))
        out = np.zeros((frames, self.channels), dtype=np.float32)
        with self._lock:
            n = min(frames, self._size)
            if n > 0:
                first = min(n, self.capacity - self._read)
                out[:first] = self._data[self._read : self._read + first]
                remaining = n - first
                if remaining > 0:
                    out[first:n] = self._data[:remaining]
                self._read = (self._read + n) % self.capacity
                self._size -= n
            underrun = n < frames
        return out, underrun


class WavRecorder:
    """Incremental 16-bit WAV writer for the whole session."""

    def __init__(self) -> None:
        self._file = None
        self.frames = 0
        self.path: Path | None = None

    def open(self, path: str | Path, sample_rate: int, channels: int = 2) -> None:
        import soundfile as sf

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = sf.SoundFile(
            str(self.path),
            mode="w",
            samplerate=int(sample_rate),
            channels=int(channels),
            subtype="PCM_16",
        )
        self.frames = 0

    def write(self, samples: np.ndarray) -> None:
        if self._file is None:
            return
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        self._file.write(np.clip(data, -1.0, 1.0))
        self.frames += data.shape[0]

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            finally:
                self._file.close()
                self._file = None


class AudioPlayer:
    """Play generated frames from the ring buffer.

    If PortAudio or an output device is unavailable, the same consumer thread
    runs against the ring buffer without opening a device.  That keeps buffer
    and underrun measurements honest in headless mode (the audio is still
    recorded to WAV by the pipeline).
    """

    def __init__(self, config: AudioConfig, ring: RingBuffer) -> None:
        self.config = config
        self.ring = ring
        self.frames_played = 0
        self.underruns = 0
        self.blocks_played = 0
        self.error: str | None = None
        self.mode = "disabled"
        self.active = False
        self.was_audible = False
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._was_underrun = False
        self._last_block_time = 0.0

    def start(self) -> None:
        self._stop_event.clear()
        if not self.config.enabled:
            # Recording-only mode: still run the consumer so buffering and
            # pacing behave exactly as they would with a device.
            self.mode = "simulated"
        elif self._try_open_stream():
            self.mode = "sounddevice"
        else:
            self.mode = "simulated"
        self.was_audible = self.mode == "sounddevice"
        self.active = True
        self._thread = threading.Thread(
            target=self._run, name="diastole-playback", daemon=True
        )
        self._thread.start()

    def _try_open_stream(self) -> bool:
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                blocksize=self.config.block_frames,
                device=self.config.device,
                latency="low",
            )
            self._stream.start()
            return True
        except Exception as exc:  # pragma: no cover - depends on host audio
            self.error = f"{type(exc).__name__}: {exc}"
            self._stream = None
            return False

    def _run(self) -> None:
        block = self.config.block_frames
        block_seconds = block / self.config.sample_rate
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            samples, underrun = self.ring.read(block)
            self.blocks_played += 1
            if underrun:
                self.underruns += 1
                # Fade into an underrun rather than emitting a hard edge.
                fade = min(64, samples.shape[0])
                if fade > 0 and not self._was_underrun:
                    samples[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
                self._was_underrun = True
            else:
                if self._was_underrun:
                    fade = min(64, samples.shape[0])
                    if fade > 0:
                        samples[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
                self._was_underrun = False

            if self._stream is not None:
                try:
                    self._stream.write(samples)
                except Exception as exc:  # pragma: no cover - device dependent
                    self.error = f"playback error: {exc}"
                    break
            else:
                # Simulated sink: consume at exactly the device rate.
                elapsed = time.monotonic() - t0
                sleep_for = block_seconds - elapsed
                if sleep_for > 0:
                    self._stop_event.wait(sleep_for)

            self.frames_played += samples.shape[0]
            self._last_block_time = time.monotonic()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.active = False

    @property
    def is_audible(self) -> bool:
        """True if this session ever opened a real output device.

        ``mode`` deliberately keeps the backend name after ``stop()`` so the
        report describes how the session played, not merely that it ended.
        """

        return self.was_audible

    def stats(self) -> dict:
        return {
            "mode": self.mode,
            "active": self.active,
            "was_audible": self.was_audible,
            "audible": self.is_audible,
            "frames_played": self.frames_played,
            "blocks_played": self.blocks_played,
            "underruns": self.underruns,
            "error": self.error,
            "device": self.config.device,
        }
