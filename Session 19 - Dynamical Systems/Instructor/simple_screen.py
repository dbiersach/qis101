#!/usr/bin/env -S uv run
"""simple_screen.py"""

import pygame
from pygame import Color


class SimpleScreen:
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

        self.screen_width = screen_size[0] - 1
        self.screen_height = screen_size[1] - 1
        self.screen_ratio = self.screen_width / self.screen_height

        self.world_rects = []
        self.world_rects.append(world_rect)
        self.calc_world_rect()

        self.is_zooming = False
        self.zoom_pos_start = None
        self.zoom_pos_stop = None
        self.zoom_surface = None
        self.zoom_rect_color = Color(zoom_rect_color)

    def __repr__(self):
        return str(
            f"SimpleScreen({self.world_min},{self.world_max},{self.screen_size}",
            f"{self.world_rects}",
        )

    def calc_world_rect(self):
        self.world_min, self.world_max = self.world_rects[-1]
        self.wx1 = self.world_min[0]
        self.wy1 = self.world_min[1]
        self.wx2 = self.world_max[0]
        self.wy2 = self.world_max[1]

        self.world_width = self.wx2 - self.wx1
        self.world_height = self.wy2 - self.wy1
        self.rw = self.screen_width / self.world_width
        self.rh = self.screen_height / self.world_height

    def screen_x(self, wx):
        return round(self.rw * (wx - self.wx1))

    def screen_y(self, wy):
        return round(self.rh * (wy - self.wy1))

    def world_x(self, sx):
        return self.wx1 + sx / self.rw

    def world_y(self, sy):
        return self.wy1 + sy / self.rh

    def set_background(self, clr):
        self.surface.fill(Color(clr))

    def set_title(self, title):
        pygame.display.set_caption(title)

    def set_screen_pixel(self, sx, sy, clr):
        if sx >= 0 and sx <= self.screen_width and sy >= 0 and sy <= self.screen_height:
            self.pixels[sx, self.screen_height - sy] = Color(clr)[:3]

    def set_world_pixel(self, wx, wy, clr):
        sx = self.screen_x(wx)
        sy = self.screen_y(wy)
        self.set_screen_pixel(sx, sy, clr)

    def flip(self):
        pygame.display.flip()

    def update(self):
        self.set_background(self.background_color)
        self.pixels = pygame.surfarray.pixels3d(self.surface)
        if self.draw_function:
            self.draw_function(self)
        self.surface.unlock()
        del self.pixels
        pygame.display.flip()

    def create_zoom_rect(self, event):
        self.zoom_pos_stop = event.pos
        zoom_width = self.zoom_pos_stop[0] - self.zoom_pos_start[0]  # type: ignore
        zoom_height = self.zoom_pos_stop[1] - self.zoom_pos_start[1]  # type: ignore
        zoom_rect = pygame.Rect((self.zoom_pos_start, (zoom_width, zoom_height)))
        zoom_rect.normalize()
        # Ensure zoom rect preserves screen aspect ratio
        zoom_rect.width = self.screen_ratio * zoom_rect.height
        return zoom_rect

    def show(self):
        # Show the screen at least once
        self.update()
        # Enter game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:  # noqa: SIM102
                    # Check if left button is being held down
                    if event.buttons[0] == 1:  # LEFT BUTTON
                        if not self.is_zooming:
                            self.is_zooming = True
                            self.zoom_pos_start = event.pos
                            self.zoom_pos_stop = None
                            self.zoom_surface = self.surface.copy()
                            # Save current image before start of zoom
                            self.zoom_surface.blit(self.surface, (0, 0))
                        else:
                            # Restore image before start of zoom
                            self.surface.blit(self.zoom_surface, (0, 0))
                            # Construct the new zoom rectangle
                            zoom_rect = self.create_zoom_rect(event)
                            # Draw the new zoom rect
                            pygame.draw.rect(
                                self.surface, self.zoom_rect_color, zoom_rect, 3
                            )
                            pygame.display.flip()
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # LEFT BUTTON  # noqa: SIM102
                        if self.is_zooming:
                            self.is_zooming = False
                            # Restore image before start of zoom
                            self.surface.blit(self.zoom_surface, (0, 0))
                            # Calculate new world rectangle
                            zoom_rect = self.create_zoom_rect(event)
                            if zoom_rect.width > 0 and zoom_rect.height > 0:
                                new_wx1 = self.world_x(zoom_rect.left)
                                new_wy1 = self.world_y(
                                    self.screen_height
                                    - (zoom_rect.top + zoom_rect.height)
                                )
                                new_wx2 = self.world_x(zoom_rect.left + zoom_rect.width)
                                new_wy2 = self.world_y(
                                    self.screen_height - zoom_rect.top
                                )
                                # Push new world rectangle onto stack
                                self.world_rects.append(
                                    ((new_wx1, new_wy1), (new_wx2, new_wy2))
                                )
                                # Redraw world using this new world rectangle
                                self.calc_world_rect()  #
                                self.update()
                    if event.button == 3:  # RIGHT BUTTON  # noqa: SIM102
                        # Restore prior zoom state
                        if len(self.world_rects) > 1:
                            self.world_rects.pop()
                            self.calc_world_rect()
                            self.update()
                # Allow client to handle events
                if self.event_function:
                    self.event_function(self, event)
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()


def main():
    print("This module is intended to be imported, not executed directly")


if __name__ == "__main__":
    main()
