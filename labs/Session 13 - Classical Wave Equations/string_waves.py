#!/usr/bin/env python3
"""string_waves.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

one_frame, two_plucks = True, False
# one_frame, two_plucks = False, False
# one_frame, two_plucks = True, True
# one_frame, two_plucks = False, True


def plot(ax):
    global pts, xa, ya_prior, ya_current, ya_next, wave

    pts = 500
    xa = np.linspace(0, 1, pts)

    ya_prior = np.exp(-1e3 * (xa - 0.3) ** 2)
    if two_plucks:
        ya_prior += np.exp(-1e2 * (xa - 0.5) ** 2)
    ya_current = np.copy(ya_prior)
    ya_next = np.zeros(pts)

    (wave,) = ax.plot(xa, ya_current, color="black", animated=True)

    ax.set_title("Wave on a String")
    ax.set_xlabel("Location")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.1, 1.1)


def anim_frame_counter():
    n = 0
    while n < 1200:  # = 30 secs @ 40 frames/sec
        n += 1
        yield n
        if one_frame:
            break


def anim_draw_frame(t):
    global ya_prior, ya_current, ya_next

    ya_prior[0], ya_prior[-1] = 0, 0
    ya_current[0], ya_current[-1] = 0, 0

    for i in range(1, pts - 1):
        ya_next[i] = ya_current[i - 1] + ya_current[i + 1] - ya_prior[i]

    ya_next[0], ya_next[-1] = 0, 0

    ya_prior = np.copy(ya_current)
    ya_current = np.copy(ya_next)

    wave.set_data(xa, ya_current)

    return (wave,)


def main():
    global anim
    plt.figure(Path(__file__).name)
    ax = plt.gca()

    plot(ax)

    anim = FuncAnimation(
        ax.figure,
        anim_draw_frame,
        anim_frame_counter,
        interval=25,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
