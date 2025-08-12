#!/usr/bin/env python3
"""logistics_map.py"""

import numpy as np
import pygame
from simple_screen import SimpleScreen


def plot_logistics_map(ss):
    for sx in range(ss.screen_width):
        x = ss.world_x(sx)
        y = np.random.random()

        # Iterate (but don't draw) to reach a stable orbit
        for i in range(500):
            y = x * y * (1 - y)

        for i in range(500):
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
    return


def main():
    ss = SimpleScreen(
        world_rect=((2.5, 0), (4.0, 1)),
        draw_function=plot_logistics_map,
        event_function=handle_events,
        screen_size=(900, 900),
        title="Logistics Map",
    )
    ss.show()


if __name__ == "__main__":
    main()
