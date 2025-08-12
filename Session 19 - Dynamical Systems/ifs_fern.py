#!/usr/bin/env python3
"""ifs_fern.py"""

import pygame
from ifs import IteratedFunctionSystem
from simple_screen import SimpleScreen

ifs = IteratedFunctionSystem()


def handle_events(ss, event):
    if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
        wr = ss.world_rects[-1]
        print(
            f"Current world rectangle: "
            f"({wr[0][0]:.4f}, {wr[0][1]:.4f}) - "
            f"({wr[1][0]:.4f}, {wr[1][1]:.4f})"
        )
    return


def plot_ifs(ss):
    iterations = 200_000
    x, y = 0, 0

    # Iterate (but don't draw) to reach a stable orbit
    for _ in range(100):
        x, y, color = ifs.transform_point(x, y)

    for _ in range(iterations):
        x, y, color = ifs.transform_point(x, y)
        ss.set_world_pixel(x, y, color)


def main():
    ifs.set_base_frame(0, 0, 48, 48)

    ifs.add_mapping(24, 0, 24, 0, 24, 10, "green", 0.01)
    ifs.add_mapping(20, 4, 28, 17.5, 6, 12, "green", 0.07)
    ifs.add_mapping(20.5, 14, 28, -1, 35, 22.5, "green", 0.07)
    ifs.add_mapping(4, 12.5, 44, 9, 7.5, 53, "green", 0.85)

    ifs.generate_transforms()

    ss = SimpleScreen(
        world_rect=((-5, -5), (55, 65)),
        screen_size=(900, 900),
        draw_function=plot_ifs,
        event_function=handle_events,
        title="Barnsley Fern",
    )
    ss.show()


if __name__ == "__main__":
    main()
