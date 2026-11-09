"""Ring buffer tests, including wrap-around and underrun."""

from __future__ import annotations

import numpy as np

from diastole.audio.playback import RingBuffer


def test_write_read_roundtrip() -> None:
    ring = RingBuffer(10, 2)
    data = np.arange(12, dtype=np.float32).reshape(6, 2)
    assert ring.write(data) == 6
    assert ring.available == 6
    out, underrun = ring.read(6)
    assert not underrun
    np.testing.assert_allclose(out, data)
    assert ring.available == 0


def test_underrun_pads_with_zeros() -> None:
    ring = RingBuffer(10, 2)
    ring.write(np.ones((3, 2), dtype=np.float32))
    out, underrun = ring.read(5)
    assert underrun
    assert np.all(out[:3] == 1.0)
    assert np.all(out[3:] == 0.0)


def test_full_buffer_partial_write() -> None:
    ring = RingBuffer(4, 2)
    assert ring.write(np.ones((6, 2), dtype=np.float32)) == 4
    assert ring.free == 0
    assert ring.write(np.ones((1, 2), dtype=np.float32)) == 0


def test_wraparound() -> None:
    ring = RingBuffer(5, 2)
    a = np.full((3, 2), 1.0, dtype=np.float32)
    b = np.full((4, 2), 2.0, dtype=np.float32)
    ring.write(a)
    out, _ = ring.read(2)
    np.testing.assert_allclose(out, a[:2])
    ring.write(b)  # wraps
    out, underrun = ring.read(5)
    assert not underrun
    np.testing.assert_allclose(out[:1], a[2:])
    np.testing.assert_allclose(out[1:], b)


def test_clear_and_fill_fraction() -> None:
    ring = RingBuffer(10, 2)
    ring.write(np.ones((5, 2), dtype=np.float32))
    assert ring.fill_fraction == 0.5
    ring.clear()
    assert ring.available == 0
    assert ring.fill_fraction == 0.0
