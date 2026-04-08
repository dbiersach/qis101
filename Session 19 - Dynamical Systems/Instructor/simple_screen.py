#!/usr/bin/env -S uv run
"""simple_screen.py

A lightweight pygame wrapper that provides a resizable world-coordinate
window with interactive mouse-drag zoom (left button) and zoom-stack
undo (right button).

Intended to be imported by IFS demos and other Session 19 scripts.
There is exactly one copy of this file, located at:

    Session 19 - Dynamical Systems/simple_screen.py

The Instructor/ subdirectory contains a symlink to this file so that
instructor scripts can use the identical unmodified class.
"""

import pygame
from pygame import Color


class SimpleScreen:
    """Pygame display surface with world-coordinate mapping and zoom support.

    Parameters
    ----------
    world_rect : tuple[tuple[float, float], tuple[float, float]]
        Initial world coordinate window as ((x_min, y_min), (x_max, y_max)).
    screen_size : tuple[int, int]
        Pixel dimensions of the display window (width, height).
    draw_function : callable or None
        Called with this SimpleScreen as its only argument each time the
        display needs to be redrawn (zoom, resize, or first show).
    event_function : callable or None
        Called with (SimpleScreen, pygame.event.Event) for every event
        that the built-in handler does not consume.
    title : str
        Window title bar text.
    background_color : str
        Pygame color name used to clear the background before each redraw.
    zoom_rect_color : str
        Color of the drag-zoom selection rectangle.
    """

    def __init__(
        self,
        world_rect,
        screen_size=(500, 500),
        draw_function=None,
        event_function=None,
        title="",
        background_color="black",
        zoom_rect_color="white",
    ):
        pygame.init()
        pygame.display.set_mode(screen_size)
        pygame.display.set_caption(title)
        self.surface = pygame.display.get_surface()
        self.background_color = Color(background_color)

        self.draw_function = draw_function
        self.event_function = event_function

        # Store the original screen_size tuple for use in __repr__
        self.screen_size = screen_size

        # screen_width / screen_height are the maximum valid pixel indices
        # (not the pixel counts), because pygame.surfarray uses 0-based indexing.
        self.screen_width = screen_size[0] - 1
        self.screen_height = screen_size[1] - 1
        self.screen_ratio = self.screen_width / self.screen_height

        self.world_rects = [world_rect]
        self.calc_world_rect()

        self.is_zooming = False
        self.zoom_pos_start = None
        self.zoom_pos_stop = None
        self.zoom_surface = None
        self.zoom_rect_color = Color(zoom_rect_color)

    def __repr__(self) -> str:
        return (
            f"SimpleScreen("
            f"world_min={self.world_min}, "
            f"world_max={self.world_max}, "
            f"screen_size={self.screen_size})"
        )

    # ── World / screen coordinate mapping ────────────────────────────────────

    def calc_world_rect(self):
        """Recompute all world-to-screen scale factors from world_rects[-1]."""
        self.world_min, self.world_max = self.world_rects[-1]
        self.wx1 = self.world_min[0]
        self.wy1 = self.world_min[1]
        self.wx2 = self.world_max[0]
        self.wy2 = self.world_max[1]

        self.world_width = self.wx2 - self.wx1
        self.world_height = self.wy2 - self.wy1
        self.rw = self.screen_width / self.world_width
        self.rh = self.screen_height / self.world_height

    def screen_x(self, wx) -> int:
        """Convert world x-coordinate to screen pixel column."""
        return round(self.rw * (wx - self.wx1))

    def screen_y(self, wy) -> int:
        """Convert world y-coordinate to screen pixel row."""
        return round(self.rh * (wy - self.wy1))

    def world_x(self, sx) -> float:
        """Convert screen pixel column to world x-coordinate."""
        return self.wx1 + sx / self.rw

    def world_y(self, sy) -> float:
        """Convert screen pixel row to world y-coordinate."""
        return self.wy1 + sy / self.rh

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def set_background(self, clr):
        """Fill the entire surface with *clr* (a pygame color name or Color)."""
        self.surface.fill(Color(clr))

    def set_title(self, title: str):
        """Update the window title bar."""
        pygame.display.set_caption(title)

    def set_screen_pixel(self, sx, sy, clr):
        """Set pixel (sx, sy) in screen coordinates.

        Out-of-bounds coordinates are silently ignored.  The y-axis is
        flipped so that world y increases upward (mathematical convention)
        while screen y increases downward (pygame convention).
        """
        if 0 <= sx <= self.screen_width and 0 <= sy <= self.screen_height:
            self.pixels[sx, self.screen_height - sy] = Color(clr)[:3]

    def set_world_pixel(self, wx, wy, clr):
        """Set the pixel at world coordinates (wx, wy)."""
        sx = self.screen_x(wx)
        sy = self.screen_y(wy)
        self.set_screen_pixel(sx, sy, clr)

    def flip(self):
        """Push the back buffer to the display (thin wrapper around pygame.display.flip)."""
        pygame.display.flip()

    def update(self):
        """Clear the surface, invoke draw_function, and flip to the display."""
        self.set_background(self.background_color)
        self.pixels = pygame.surfarray.pixels3d(self.surface)
        if self.draw_function:
            self.draw_function(self)
        self.surface.unlock()
        del self.pixels
        pygame.display.flip()

    # ── Zoom helpers ──────────────────────────────────────────────────────────

    def create_zoom_rect(self, event) -> pygame.Rect:
        """Build a normalized, aspect-ratio-preserving zoom rectangle."""
        self.zoom_pos_stop = event.pos
        zoom_width = self.zoom_pos_stop[0] - self.zoom_pos_start[0]  # type: ignore[index]
        zoom_height = self.zoom_pos_stop[1] - self.zoom_pos_start[1]  # type: ignore[index]
        zoom_rect = pygame.Rect((self.zoom_pos_start, (zoom_width, zoom_height)))
        zoom_rect.normalize()
        # Constrain width to preserve the screen aspect ratio
        zoom_rect.width = round(self.screen_ratio * zoom_rect.height)
        return zoom_rect

    # ── Main event loop ───────────────────────────────────────────────────────

    def show(self):
        """Enter the pygame event loop.

        Left-button drag: draw a zoom rectangle and zoom into it on release.
        Right-button click: pop the zoom stack, restoring the previous view.
        Any other events are passed to event_function (if provided).
        """
        # Draw at least once before waiting for events
        self.update()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:  # noqa: SIM102
                    if event.buttons[0] == 1:  # left button held
                        if not self.is_zooming:
                            self.is_zooming = True
                            self.zoom_pos_start = event.pos
                            self.zoom_pos_stop = None
                            # Snapshot the current image so we can restore it
                            # while the user drags out the selection rectangle
                            self.zoom_surface = self.surface.copy()
                            self.zoom_surface.blit(self.surface, (0, 0))
                        else:
                            # Redraw the snapshot, then overlay the current rect
                            self.surface.blit(self.zoom_surface, (0, 0))
                            zoom_rect = self.create_zoom_rect(event)
                            pygame.draw.rect(
                                self.surface, self.zoom_rect_color, zoom_rect, 3
                            )
                            pygame.display.flip()

                if event.type == pygame.MOUSEBUTTONUP:
                    if (
                        event.button == 1 and self.is_zooming
                    ):  # left button released  # noqa: SIM102
                        self.is_zooming = False
                        self.surface.blit(self.zoom_surface, (0, 0))
                        zoom_rect = self.create_zoom_rect(event)
                        if zoom_rect.width > 0 and zoom_rect.height > 0:
                            new_wx1 = self.world_x(zoom_rect.left)
                            new_wy1 = self.world_y(
                                self.screen_height - (zoom_rect.top + zoom_rect.height)
                            )
                            new_wx2 = self.world_x(zoom_rect.left + zoom_rect.width)
                            new_wy2 = self.world_y(self.screen_height - zoom_rect.top)
                            self.world_rects.append(
                                ((new_wx1, new_wy1), (new_wx2, new_wy2))
                            )
                            self.calc_world_rect()
                            self.update()

                    if (
                        event.button == 3 and len(self.world_rects) > 1
                    ):  # right button: pop zoom stack
                        self.world_rects.pop()
                        self.calc_world_rect()
                        self.update()

                # Let the caller handle any events we did not consume
                if self.event_function:
                    self.event_function(self, event)

                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()


def main():
    print("This module is intended to be imported, not executed directly.")


if __name__ == "__main__":
    main()
