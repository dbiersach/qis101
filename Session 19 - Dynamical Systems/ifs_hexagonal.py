#!/usr/bin/env python3
"""ifs_hexagonal.py"""

import numpy as np
from ifs import IteratedFunctionSystem
from pygame import Color
from simple_screen import SimpleScreen

ifs = IteratedFunctionSystem()


def plot_ifs(ss):
    iterations = 200_000
    x, y = 0.0, 0.0

    # Iterate 100 times, but don't draw any pixels
    # This will give the IFS "time" reach its stable orbit
    for _ in range(100):
        x, y, clr = ifs.transform_point(x, y)

    # Now draw each pixel in the stable orbit
    for _ in range(iterations):
        x, y, clr = ifs.transform_point(x, y)
        ss.set_world_pixel(x, y, clr)


def main():
    ifs.set_base_frame(0, 0, 30, 30)

    h = 5 * np.sqrt(3)

    p = 1 / 6

    # Enter your mappings here

    # COD
    ifs.add_mapping(25, 15, 15, 15, 20, 15 + h, Color("blue"), p)
    # DOE

    # EOF

    # FOA

    # AOB

    # BOC

    ifs.generate_transforms()

    ss = SimpleScreen(
        world_rect=((0, 0), (30, 30)),
        screen_size=(900, 900),
        draw_function=plot_ifs,
        title="IFS Hexagonal",
    )
    ss.show()


if __name__ == "__main__":
    main()
