"""Sensor abstractions: physiological samples and heart-rate sources."""

from __future__ import annotations

import abc
import dataclasses
import enum
import threading
import time


class SensorStatus(str, enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class PhysiologicalSample:
    """One instantaneous heart-rate reading."""

    bpm: float
    timestamp: float
    valid: bool = True
    source: str = "simulated"
    note: str = ""

    @property
    def age(self) -> float:
        return 0.0


class HeartRateSensor(abc.ABC):
    """Common interface for simulated and real sensors.

    Implementations must be safe to call from the generation thread while their
    own acquisition thread writes.  ``read`` never blocks for more than a lock
    acquisition.
    """

    def __init__(self) -> None:
        self._status = SensorStatus.DISCONNECTED
        self._status_lock = threading.Lock()

    @property
    def status(self) -> SensorStatus:
        with self._status_lock:
            return self._status

    def _set_status(self, status: SensorStatus) -> None:
        with self._status_lock:
            self._status = status

    @abc.abstractmethod
    def start(self) -> None:
        """Begin acquisition."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop acquisition and release resources."""

    @abc.abstractmethod
    def read(self) -> PhysiologicalSample | None:
        """Return the most recent sample (or None before the first sample)."""

    @abc.abstractmethod
    def set_manual_bpm(self, bpm: float | None) -> None:
        """Override the source with a fixed BPM; None returns to the scenario."""

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "HeartRateSensor":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class StaticSensor(HeartRateSensor):
    """Deterministic sensor for tests and headless experiments."""

    def __init__(self, bpm: float = 62.0) -> None:
        super().__init__()
        self._bpm = bpm
        self._lock = threading.Lock()
        self._sample = PhysiologicalSample(bpm=bpm, timestamp=time.monotonic(), source="static")
        self._set_status(SensorStatus.CONNECTED)

    def start(self) -> None:
        self._set_status(SensorStatus.CONNECTED)

    def stop(self) -> None:
        self._set_status(SensorStatus.DISCONNECTED)

    def read(self) -> PhysiologicalSample:
        with self._lock:
            return self._sample

    def set_manual_bpm(self, bpm: float | None) -> None:
        if bpm is None:
            return
        with self._lock:
            self._bpm = float(bpm)
            self._sample = PhysiologicalSample(
                bpm=float(bpm), timestamp=time.monotonic(), source="static"
            )
