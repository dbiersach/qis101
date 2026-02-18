#!/usr/bin/env -S uv run
"""ifs_bnl.py"""

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
    iterations = 300_000
    x, y = 0, 0

    # Iterate (but don't draw) to let IFS reach its stable orbit
    for _ in range(100):
        x, y, color = ifs.transform_point(x, y)

    for _ in range(iterations):
        x, y, color = ifs.transform_point(x, y)
        ss.set_world_pixel(x, y, color)


def main():
    ifs.set_base_frame(0, 0, 17, 7)

    p = 1 / 29

    ifs.add_mapping(0, 6, 2, 6, 0, 7, "lightblue", p)  # 01
    ifs.add_mapping(2, 6, 4, 6, 2, 7, "blue", p)  # 02
    ifs.add_mapping(1, 4, 1, 6, 0, 4, "blue", p)  # 03
    ifs.add_mapping(4, 6, 4, 4, 5, 6, "blue", p)  # 04
    ifs.add_mapping(0, 3, 2, 3, 0, 4, "blue", p)  # 05
    ifs.add_mapping(4, 4, 2, 4, 4, 3, "red", p)  # 06
    ifs.add_mapping(1, 1, 1, 3, 0, 1, "pink", p)  # 07
    ifs.add_mapping(4, 3, 4, 1, 5, 3, "blue", p)  # 08
    ifs.add_mapping(0, 0, 2, 0, 0, 1, "blue", p)  # 09
    ifs.add_mapping(4, 1, 2, 1, 4, 0, "sienna", p)  # 10

    ifs.add_mapping(6, 6, 7, 6, 6, 7, "blue", p)  # 11
    ifs.add_mapping(10, 6, 11, 6, 10, 7, "papayawhip", p)  # 12
    ifs.add_mapping(6, 5, 8, 5, 6, 6, "chartreuse", p)  # 13
    ifs.add_mapping(11, 4, 11, 6, 10, 4, "blue", p)  # 14
    ifs.add_mapping(7, 3, 7, 5, 6, 3, "blue", p)  # 15
    ifs.add_mapping(8, 4, 9, 4, 8, 5, "blue", p)  # 16
    ifs.add_mapping(7, 1, 7, 3, 6, 1, "orange", p)  # 17
    ifs.add_mapping(9, 2, 9, 4, 8, 2, "blue", p)  # 18
    ifs.add_mapping(10, 4, 10, 2, 11, 4, "maroon", p)  # 19
    ifs.add_mapping(6, 0, 7, 0, 6, 1, "blue", p)  # 20
    ifs.add_mapping(9, 1, 11, 1, 9, 2, "blue", p)  # 21
    ifs.add_mapping(10, 0, 11, 0, 10, 1, "yellow", p)  # 22

    ifs.add_mapping(13, 5, 13, 7, 12, 5, "green", p)  # 23
    ifs.add_mapping(12, 4, 13, 4, 12, 5, "blue", p)  # 24
    ifs.add_mapping(12, 4, 12, 2, 13, 4, "blue", p)  # 25
    ifs.add_mapping(13, 1, 13, 2, 12, 1, "white", p)  # 26
    ifs.add_mapping(12, 0, 14, 0, 12, 1, "blue", p)  # 27
    ifs.add_mapping(15, 0, 15, 1, 14, 0, "blue", p)  # 28
    ifs.add_mapping(15, 0, 17, 0, 15, 1, "purple", p)  # 29

    ifs.generate_transforms()

    ss = SimpleScreen(
        world_rect=((0, 0), (17, 7)),
        screen_size=(1200, 500),
        draw_function=plot_ifs,
        event_function=handle_events,
        title="Brookhaven National Laboratory",
    )
    ss.show()


if __name__ == "__main__":
    main()
