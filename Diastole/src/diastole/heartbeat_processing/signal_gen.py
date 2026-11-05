import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Configuration
SAMPLE_RATE = 100
WINDOW = 5
BPM = 75

buffer_size = SAMPLE_RATE * WINDOW
signal = np.zeros(buffer_size)
time = np.arange(buffer_size) / SAMPLE_RATE

phase = 0.0
elapsed = 0.0
current_bpm = BPM

def generate_ppg(phase):
    systolic = np.exp(-((phase - 0.15) / 0.055) ** 2)
    diastolic = 0.25 * np.exp(-((phase - 0.38) / 0.09) ** 2)
    return systolic + diastolic

fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(time, signal, color="lime", lw=1.5)

ax.set_ylim(-0.2, 1.3)
ax.set_xlim(0, WINDOW)
ax.set_facecolor("#101010")
fig.patch.set_facecolor("#101010")
ax.tick_params(colors="white")
ax.grid(color="#333333", alpha=0.5)

title = ax.set_title(f"Pulseform | {BPM} BPM", color="white")

def update(frame):
    global phase, elapsed, current_bpm, signal

    # Change BPM periodically
    if frame % 200 == 0:
        current_bpm = random.randint(60, 100)

    # Generate one new sample
    phase = (phase + current_bpm / (60 * SAMPLE_RATE)) % 1
    new_sample = generate_ppg(phase)

    # Shift buffer
    signal[:-1] = signal[1:]
    signal[-1] = new_sample
    elapsed += 1 / SAMPLE_RATE

    # Update display
    line.set_ydata(signal)
    line.set_xdata(np.arange(buffer_size) / SAMPLE_RATE + elapsed - WINDOW)
    ax.set_xlim(elapsed - WINDOW, elapsed)

    title.set_text(f"Pulseform | {current_bpm} BPM")

    return line, title

ani = FuncAnimation(
    fig, update,
    interval=1000 / SAMPLE_RATE,
    blit=True,
    cache_frame_data=False
)

plt.show()