"""Simulated heart-rate sensor.

The simulator is intentionally more than a sine wave: it combines
  * a scenario profile (rest / gradual / sudden / workout / demo / random),
  * Ornstein-Uhlenbeck noise (short-term physiological jitter),
  * respiratory sinus arrhythmia (a 0.25 Hz oscillation),
  * an optional manual override for interactive control.

Every sample is timestamped with ``time.monotonic`` so the processing layer can
detect staleness independently of the sensor's own clock.
"""

from __future__ import annotations

import math
import random
import threading
import time

from diastole.sensors.base import (
    HeartRateSensor,
    PhysiologicalSample,
    SensorStatus,
)

SCENARIOS = ("rest", "gradual", "sudden", "workout", "demo", "random", "manual")


def _smoothstep(lo: float, hi: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def scenario_profile(name: str, t: float, base_bpm: float = 62.0) -> float:
    """Return the deterministic centre BPM for a scenario at elapsed time ``t``."""

    if name == "rest":
        return base_bpm
    if name == "gradual":
        return base_bpm + min(85.0, 1.2 * t)
    if name == "sudden":
        return base_bpm if t < 8.0 else base_bpm + 80.0
    if name == "workout":
        # Warm-up, plateau, cool-down over 60 seconds.
        if t < 10.0:
            return base_bpm + 3.0 * t
        if t < 40.0:
            return base_bpm + 30.0 + min(45.0, 1.5 * (t - 10.0))
        return max(base_bpm, base_bpm + 75.0 - 2.5 * (t - 40.0))
    if name == "demo":
        # A narrated arc: rest -> tachycardia -> recovery -> rest.
        if t < 8.0:
            return 62.0
        if t < 14.0:
            return 62.0 + (132.0 - 62.0) * _smoothstep(8.0, 14.0, t)
        if t < 26.0:
            return 132.0
        if t < 34.0:
            return 132.0 + (64.0 - 132.0) * _smoothstep(26.0, 34.0, t)
        return 64.0
    if name == "manual":
        return base_bpm
    if name == "random":
        # Slow random walk; the OU process supplies the fast component.
        return base_bpm + 35.0 * math.sin(t / 11.0) + 12.0 * math.sin(t / 3.7)
    raise ValueError(f"Unknown scenario {name!r}; expected one of {SCENARIOS}")


class SimulatedHeartRateSensor(HeartRateSensor):
    """A thread-driven, reproducible simulated heart-rate source."""

    def __init__(
        self,
        scenario: str = "rest",
        base_bpm: float = 62.0,
        update_hz: float = 4.0,
        seed: int = 0,
        noise_sigma: float = 1.4,
        ou_theta: float = 0.7,
        rsa_amplitude: float = 2.5,
        rsa_hz: float = 0.25,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")
        super().__init__()
        self.scenario = scenario
        self.base_bpm = float(base_bpm)
        self.update_hz = float(update_hz)
        self.seed = int(seed)
        self.noise_sigma = float(noise_sigma)
        self.ou_theta = float(ou_theta)
        self.rsa_amplitude = float(rsa_amplitude)
        self.rsa_hz = float(rsa_hz)

        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._latest: PhysiologicalSample | None = None
        self._manual_bpm: float | None = None
        self._scenario_started = time.monotonic()
        self._session_started = self._scenario_started
        self._ou_state = 0.0
        self._rsa_phase = self._rng.random() * math.tau
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._sample_count = 0

    # -- public API -------------------------------------------------------
    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._sample_count

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._set_status(SensorStatus.CONNECTING)
        self._stop_event.clear()
        self._session_started = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="diastole-sensor", daemon=True
        )
        self._thread.start()
        self._set_status(SensorStatus.CONNECTED)

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        self._set_status(SensorStatus.DISCONNECTED)

    def read(self) -> PhysiologicalSample | None:
        with self._lock:
            return self._latest

    def set_manual_bpm(self, bpm: float | None) -> None:
        with self._lock:
            if bpm is not None:
                bpm = float(bpm)
                if not (30.0 <= bpm <= 220.0):
                    raise ValueError("Manual BPM must be between 30 and 220")
            self._manual_bpm = bpm

    def set_scenario(self, scenario: str) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")
        with self._lock:
            self.scenario = scenario
            self._scenario_started = time.monotonic()
            self._manual_bpm = None

    def reset_clock(self, now: float | None = None) -> None:
        """Restart elapsed time without restarting the thread (used by tests)."""

        with self._lock:
            self._scenario_started = now if now is not None else time.monotonic()
            self._session_started = self._scenario_started
            self._ou_state = 0.0

    # -- simulation -------------------------------------------------------
    def bpm_at(self, elapsed: float) -> float:
        """Pure scenario value at an elapsed time, including RSA (no noise)."""

        with self._lock:
            scenario = self.scenario
            manual = self._manual_bpm
            phase = self._rsa_phase
            base = self.base_bpm
        if manual is not None:
            return manual
        centre = scenario_profile(scenario, elapsed, base)
        rsa = self.rsa_amplitude * math.sin(math.tau * self.rsa_hz * elapsed + phase)
        rsa *= max(0.3, centre / 70.0)
        return max(30.0, min(220.0, centre + rsa))

    def _next_sample(self, now: float, dt: float) -> PhysiologicalSample:
        with self._lock:
            manual = self._manual_bpm
            scenario = self.scenario
            started = self._scenario_started
            elapsed = now - started

        if manual is not None:
            bpm = manual
            source = "manual"
            valid = True
            note = "manual override"
        else:
            # Ornstein-Uhlenbeck jitter is updated under the sample loop.
            shock = self._rng.gauss(0.0, 1.0)
            self._ou_state += (
                -self.ou_theta * self._ou_state * dt
                + self.noise_sigma * math.sqrt(max(dt, 1e-6)) * shock
            )
            self._ou_state = max(-12.0, min(12.0, self._ou_state))
            centre = scenario_profile(scenario, elapsed, self.base_bpm)
            rsa = self.rsa_amplitude * math.sin(
                math.tau * self.rsa_hz * elapsed + self._rsa_phase
            )
            rsa *= max(0.3, centre / 70.0)
            bpm = max(30.0, min(220.0, centre + rsa + self._ou_state))
            source = "simulated"
            valid = True
            note = scenario

        return PhysiologicalSample(
            bpm=float(bpm), timestamp=now, valid=valid, source=source, note=note
        )

    def _run(self) -> None:
        period = 1.0 / self.update_hz
        last = time.monotonic()
        next_tick = last
        while not self._stop_event.is_set():
            now = time.monotonic()
            dt = now - last
            last = now
            sample = self._next_sample(now, dt)
            with self._lock:
                self._latest = sample
                self._sample_count += 1
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                # Do not accumulate a backlog after a scheduling hiccup.
                next_tick = time.monotonic()
