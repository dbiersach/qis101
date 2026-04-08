#!/usr/bin/env -S uv run
"""logistics_map.py"""

import numpy as np
import pygame
from simple_screen import SimpleScreen


def plot_logistics_map(ss):
    for sx in range(ss.screen_width):
        x = ss.world_x(sx)
        y = np.random.random()

        # Warmup phase: iterate the logistic map without plotting to allow
        # transient behavior to die out and the trajectory to settle onto
        # its long-term attractor (fixed point, cycle, or chaotic band)
        for _ in range(500):
            y = x * y * (1 - y)

        # Drawing phase: iterate again now that the orbit is stable and
        # plot each point — this reveals the attractor's structure for
        # this value of the growth parameter x
        for _ in range(500):
            y = x * y * (1 - y)
            ss.set_world_pixel(x, y, "blue")


def handle_events(ss, event):
    if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
        wr = ss.world_rects[-1]
        print(
            f"Current world rectangle: "
            f"({wr[0][0]:.4f}, {wr[0][1]:.4f}) - "
            f"({wr[1][0]:.4f}, {wr[1][1]:.4f})"
        )


def main():
    ss = SimpleScreen(
        world_rect=((2.5, 0), (4.0, 1)),
        draw_function=plot_logistics_map,
        event_function=handle_events,
        screen_size=(800, 800),
        title="Logistics Map",
    )
    ss.show()


if __name__ == "__main__":
    main()
