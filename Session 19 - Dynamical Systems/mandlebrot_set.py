#!/usr/bin/env python3
"""mandlebrot_set.py"""

import pygame
from pygame import Color
from simple_screen import SimpleScreen


def plot_mandlebrot_set(ss):
    max_iter = 100
    radius = 16

    for sy in reversed(range(ss.screen_height)):
        wy = ss.world_y(sy)
        for sx in range(ss.screen_width):
            wx = ss.world_x(sx)
            zx, zy = wx, wy  # Initialize z = C (real = wx, imag = wy)
            zx_2, zy_2 = zx * zx, zy * zy  # Square real and imaginary components
            iter = 0
            while zx_2 + zy_2 < radius and iter < max_iter:
                nx = zx_2 - zy_2 + wx  # real part
                ny = 2 * zx * zy + wy  # imag part
                zx = nx
                zy = ny
                zx_2 = zx * zx
                zy_2 = zy * zy
                iter += 1
            # Select color using HSV encoding
            hue = int(360 * iter / max_iter)
            saturation = 100
            value = 100 if iter < max_iter else 0
            clr = Color(0)
            clr.hsva = (hue, saturation, value)
            ss.set_screen_pixel(sx, sy, (clr.r, clr.g, clr.b))
        ss.flip()


def handle_events(ss, event):
    if event.type == pygame.KEYDOWN:  # noqa: SIM102
        if event.key == pygame.K_w:
            wr = ss.world_rects[-1]
            print(
                f"Current world rectangle: "
                f"({wr[0][0]:.4f}, {wr[0][1]:.4f}) - "
                f"({wr[1][0]:.4f}, {wr[1][1]:.4f})"
            )
    return


def main():
    ss = SimpleScreen(
        world_rect=((-2.2, -1.51), (1, 1.51)),
        screen_size=(900, 900),
        draw_function=plot_mandlebrot_set,
        event_function=handle_events,
        title="Mandelbrot Set",
    )

    ss.show()


if __name__ == "__main__":
    main()
