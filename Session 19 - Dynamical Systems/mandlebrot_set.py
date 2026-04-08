#!/usr/bin/env -S uv run
"""mandelbrot_set.py

Renders the Mandelbrot set using pygame for display and NumPy vectorized
arithmetic for performance.  The escape-radius test compares squared
magnitudes (|z|² < radius) to avoid an unnecessary square root each iteration.

Press W to print the current world-coordinate rectangle to the console.
"""

import numpy as np
import pygame
from pygame import Color
from simple_screen import SimpleScreen


def plot_mandelbrot_set(ss):
    """Draw the Mandelbrot set onto *ss* using vectorized NumPy iteration.

    Each pixel maps to a complex constant C.  The iteration z ← z² + C is
    applied up to *max_iter* times; pixels that never escape the disc of
    squared radius *radius* are colored black, while escapees are colored
    by their escape-iteration count using HSV encoding.

    Parameters
    ----------
    ss : SimpleScreen
        The target screen object, which supplies world↔screen coordinate
        conversion and pixel-setting helpers.
    """
    max_iter = 100
    radius = 16  # squared escape radius (|z|² threshold)

    # ------------------------------------------------------------------
    # Build a 2-D grid of complex constants C, one per pixel.
    # wx shape: (screen_width,)   wy shape: (screen_height,)
    # C shape:  (screen_height, screen_width)
    # ------------------------------------------------------------------
    wx = np.array([ss.world_x(sx) for sx in range(ss.screen_width)])
    wy = np.array([ss.world_y(sy) for sy in range(ss.screen_height)])
    C = wx[np.newaxis, :] + 1j * wy[:, np.newaxis]

    # Initialize Z to C so the first iteration computes Z = C² + C, which is
    # the standard Mandelbrot iteration.  Also prepare arrays to track escape
    # status and iteration counts for each pixel.
    Z = C.copy()
    iter_count = np.zeros(C.shape, dtype=int)
    escaped = np.zeros(C.shape, dtype=bool)

    # Switch to the OS wait cursor before the heavy computation begins,
    # then pump the event queue so the OS has a chance to render it
    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_WAIT)
    pygame.event.pump()

    # Vectorized Mandelbrot iteration with boolean-mask early exit
    for _ in range(max_iter):
        active = ~escaped  # pixels still being iterated (not yet escaped)
        Z[active] = Z[active] ** 2 + C[active]  # Mandelbrot iteration
        newly_escaped = active & (Z.real**2 + Z.imag**2 >= radius)  # escape test
        escaped |= newly_escaped  # update escape status for pixels that just escaped
        # Only increment the count for pixels still inside the set
        iter_count[active & ~newly_escaped] += 1

    # ------------------------------------------------------------------
    # Map iteration counts to HSV colors and push pixels to the screen.
    # Pixels that never escaped (iter_count == max_iter) receive value=0
    # (black); all others are colored by hue derived from escape speed.
    # ------------------------------------------------------------------
    hue_grid = (360 * iter_count / max_iter).astype(int)

    # Write all pixels into the off-screen buffer first, then flip once
    # so the display updates atomically rather than row-by-row.
    for sy in range(ss.screen_height):
        for sx in range(ss.screen_width):
            n = iter_count[sy, sx]
            hue = int(hue_grid[sy, sx])
            value = 0 if n == max_iter else 100
            clr = Color(0)
            clr.hsva = (hue, 100, value)
            ss.set_screen_pixel(sx, sy, (clr.r, clr.g, clr.b))
    ss.flip()
    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)


def handle_events(ss, event):
    """Print the current world-coordinate rectangle when W is pressed."""
    if event.type == pygame.KEYDOWN:  # noqa: SIM102
        if event.key == pygame.K_w:
            wr = ss.world_rects[-1]
            print(
                f"Current world rectangle: "
                f"({wr[0][0]:.4f}, {wr[0][1]:.4f}) - "
                f"({wr[1][0]:.4f}, {wr[1][1]:.4f})"
            )


def main():
    ss = SimpleScreen(
        world_rect=((-2.2, -1.51), (1, 1.51)),
        screen_size=(800, 800),
        draw_function=plot_mandelbrot_set,
        event_function=handle_events,
        title="Mandelbrot Set",
    )
    ss.show()


if __name__ == "__main__":
    main()
