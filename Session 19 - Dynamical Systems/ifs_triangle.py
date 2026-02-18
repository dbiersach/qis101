#!/usr/bin/env -S uv run
"""ifs_triangle.py"""

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

    # Iterate (but don't draw) to let IFS reach its stable orbit
    for _ in range(100):
        x, y, color = ifs.transform_point(x, y)

    for _ in range(iterations):
        x, y, color = ifs.transform_point(x, y)
        ss.set_world_pixel(x, y, color)


def main():
    ifs.set_base_frame(0, 0, 30, 30)

    p = 1 / 3
    ifs.add_mapping(0, 0, 15, 0, 0, 15, "blue", p)
    ifs.add_mapping(15, 0, 30, 0, 15, 15, "blue", p)
    ifs.add_mapping(7.5, 15, 22.5, 15, 7.5, 30, "blue", p)

    ifs.generate_transforms()

    ss = SimpleScreen(
        world_rect=((0, 0), (30, 30)),
        screen_size=(900, 900),
        draw_function=plot_ifs,
        event_function=handle_events,
        title="Sierpinksi Triangle",
    )
    ss.show()


if __name__ == "__main__":
    main()
