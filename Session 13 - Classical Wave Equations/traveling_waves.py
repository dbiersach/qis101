#!/usr/bin/env python3
"""traveling_waves.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

run_number = 0

# Each wave parameter tuple contains these values in order:
# Amplitude (amp), Wave Number (k), Angular Velocity (w)
wave_params = (
    # Run 0 (Blue wave parameters are fixed)
    (1, 1, 1 / 16),
    # Run 1 (RED wave has zero amplitude)
    (0, 0, 0),
    # Run 2 (RED wave has half amplitude)
    (1 / 2, 1, 1 / 16),
    # Run 3 (RED wave has half wave number)
    (1, 1 / 2, 1 / 16),
    # Run 4 (RED wave has half velocity)
    (1, 1, 1 / 8),
    # Run 5 (RED wave 2 has opposite velocity)
    (1, 1, -1 / 16),
    # Run 6 (only draw average of Blue and Red waves)
    (1, 1, -1 / 16),
)
amp1, k1, w1 = wave_params[0]
amp2, k2, w2 = wave_params[run_number]

t = 0  # Start time = 0 secs
x = np.linspace(0, 6 * np.pi, 600)
y1 = amp1 * np.sin(k1 * x + w1 * t)
y2 = amp2 * np.sin(k2 * x + w2 * t)
y3 = (y1 + y2) / 2  # Average of y1 and y2


def anim_frame_counter():
    n = 0
    # Animation interval set to 25 ms / frame
    # (1 frame / 25 ms) * (1000 ms / sec) = 40 frames / sec
    # (600 total frames) / (40 frames/sec) = 15 total secs
    while n < 600:
        n += 1
        yield n


def anim_draw_frame(t):
    global wave1, wave2, wave3
    y1 = amp1 * np.sin(k1 * x + w1 * t)
    y2 = amp2 * np.sin(k2 * x + w2 * t)
    y3 = (y1 + y2) / 2  # Average of y1 and y2
    wave1.set_data(x, y1)
    wave2.set_data(x, y2)
    wave3.set_data(x, y3)
    return wave1, wave2, wave3


def main():
    global wave1, wave2, wave3, anim
    plt.figure(Path(__file__).name)

    if run_number == 0:
        (wave1,) = plt.plot(x, y1, color="blue")
        (wave2,) = plt.plot(x, y2, color="white")
        (wave3,) = plt.plot(x, y3, color="blue")
    else:
        if run_number < 6:
            (wave1,) = plt.plot(x, y1, color="blue")
            (wave2,) = plt.plot(x, y2, color="red")
            (wave3,) = plt.plot(x, y3, color="black")
        else:
            # Do not show wave1 and wave2 for run #6
            (wave1,) = plt.plot(x, y1, color="white")
            (wave2,) = plt.plot(x, y2, color="white")
            (wave3,) = plt.plot(x, y3, color="black")

    plt.title(f"Traveling Waves (Run #{run_number})")
    plt.xlabel("Location")
    plt.ylabel("Amplitude")

    # fmt: off
    anim = FuncAnimation(plt.gcf(),
        anim_draw_frame, anim_frame_counter, interval=25,
        cache_frame_data=False, blit=True, repeat=False)
    # fmt: on

    plt.show()


if __name__ == "__main__":
    main()
