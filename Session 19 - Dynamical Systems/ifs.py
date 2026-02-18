#!/usr/bin/env -S uv run
"""ifs.py"""

import numpy as np


class Transform:
    def __init__(self):
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        self.x3 = None
        self.y3 = None
        self.color = None
        self.probability = None
        self.m = None

    def __repr__(self):
        return str(f"Transform({self.x1})")


class IteratedFunctionSystem:
    def __init__(self):
        self.transforms = []
        self.affine_width = None
        self.affine_height = None
        self.cdf = 0.0

    def __repr__(self):
        return str(f"IteratedFunctionSystem({self.transforms}")

    def set_base_frame(self, x_min, y_min, x_max, y_max):
        self.affine_width = x_max - x_min
        self.affine_height = y_max - y_min
        return

    def add_mapping(
        self, x_left, y_left, x_right, y_right, x_top, y_top, color, probability
    ):
        self.cdf += probability

        t = Transform()
        t.x1 = x_left
        t.y1 = y_left
        t.x2 = x_right
        t.y2 = y_right
        t.x3 = x_top
        t.y3 = y_top
        t.color = color
        t.probability = self.cdf
        self.transforms.append(t)

    def generate_transforms(self):
        for t in self.transforms:
            t.m = np.zeros((3, 3), dtype=float)

            coeffs = np.array(
                [[0, 0, 1], [self.affine_width, 0, 1], [0, self.affine_height, 1]]
            )

            vals = np.array([t.x1, t.x2, t.x3])
            sol = np.linalg.solve(coeffs, vals)
            t.m[0, 0] = sol[0]
            t.m[1, 0] = sol[1]
            t.m[2, 0] = sol[2]

            vals = np.array([t.y1, t.y2, t.y3])
            sol = np.linalg.solve(coeffs, vals)
            t.m[0, 1] = sol[0]
            t.m[1, 1] = sol[1]
            t.m[2, 1] = sol[2]

            t.m[0, 2] = 0
            t.m[1, 2] = 0
            t.m[2, 2] = 1

    def transform_point(self, x, y):
        p = np.random.random()
        for t in self.transforms:
            if p < t.probability:
                xt = x * t.m[0, 0] + y * t.m[1, 0] + t.m[2, 0]
                yt = x * t.m[0, 1] + y * t.m[1, 1] + t.m[2, 1]
                return xt, yt, t.color
        return 0, 0, 0


def main():
    print("This module is intended to be imported, not executed directly")


if __name__ == "__main__":
    main()
