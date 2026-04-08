#!/usr/bin/env -S uv run
"""ifs.py

Iterated Function System (IFS) engine used by the Session 19 fractal demos.

An IFS is a finite collection of affine transformations, each applied with a
given probability.  Iterating the random application of these transforms
from any starting point converges to the unique fractal attractor of the
system (the Collage Theorem guarantees this for contractive maps).

Classes
-------
Transform
    Stores the triangle vertices and solved affine matrix for one IFS mapping.
IteratedFunctionSystem
    Builds the affine matrices from user-supplied triangle corners and
    generates points on the attractor via the chaos game algorithm.
"""

import numpy as np


class Transform:
    """One affine mapping within an IFS, defined by three control points.

    Attributes
    ----------
    x1, y1 : float
        Left control-point coordinates.
    x2, y2 : float
        Right control-point coordinates.
    x3, y3 : float
        Top control-point coordinates.
    color : str or pygame.Color
        Color used when drawing pixels produced by this transform.
    probability : float
        Cumulative probability threshold (CDF value) for random selection.
    m : ndarray, shape (3, 3)
        Solved affine transformation matrix (set by generate_transforms).
    """

    def __init__(self):
        self.x1: float = 0.0
        self.y1: float = 0.0
        self.x2: float = 0.0
        self.y2: float = 0.0
        self.x3: float = 0.0
        self.y3: float = 0.0
        self.color = None
        self.probability: float = 0.0
        self.m: np.ndarray = np.zeros((3, 3), dtype=float)

    def __repr__(self) -> str:
        return f"Transform(x1={self.x1})"


class IteratedFunctionSystem:
    """IFS engine: builds affine maps and generates attractor points.

    Usage
    -----
    1. Call set_base_frame to declare the source triangle's bounding box.
    2. Call add_mapping once per transform, supplying the three destination
       triangle vertices, a color, and a probability weight.
    3. Call generate_transforms to solve the affine matrices.
    4. Call transform_point in a loop (the "chaos game") to generate pixels.

    The probabilities passed to add_mapping must sum to 1.0; if they do not,
    transform_point will occasionally fall through all branches and return
    the sentinel (0, 0, 0).
    """

    def __init__(self):
        self.transforms: list[Transform] = []
        self.affine_width: float | None = None
        self.affine_height: float | None = None
        self.cdf: float = 0.0
        self._rng = np.random.default_rng()

    def __repr__(self) -> str:
        return f"IteratedFunctionSystem(transforms={self.transforms})"

    def set_base_frame(self, x_min: float, y_min: float, x_max: float, y_max: float):
        """Define the source-space bounding box for all IFS mappings.

        Parameters
        ----------
        x_min, y_min : float
            Bottom-left corner of the source frame.
        x_max, y_max : float
            Top-right corner of the source frame.
        """
        self.affine_width = x_max - x_min
        self.affine_height = y_max - y_min

    def add_mapping(
        self,
        x_left: float,
        y_left: float,
        x_right: float,
        y_right: float,
        x_top: float,
        y_top: float,
        color,
        probability: float,
    ):
        """Register one affine mapping.

        The three (x, y) pairs define where the source triangle's left,
        right, and top vertices map to in world space.  The probability
        weights must sum to 1.0 across all mappings.

        Parameters
        ----------
        x_left, y_left : float
            Destination of the source frame's left vertex.
        x_right, y_right : float
            Destination of the source frame's right vertex.
        x_top, y_top : float
            Destination of the source frame's top vertex.
        color : str or pygame.Color
            Pixel color produced by this transform.
        probability : float
            Fraction of iterations assigned to this transform (0 < p ≤ 1).
            Internally stored as a running CDF.
        """
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
        """Solve the affine matrix for every registered mapping.

        For each Transform, solve the linear system

            [0,             0,              1]   [a]   [x_dest]
            [affine_width,  0,              1] · [b] = [x_dest]
            [0,             affine_height,  1]   [c]   [x_dest]

        separately for the x and y destination coordinates, then assemble
        the result into the 3×3 homogeneous matrix t.m.
        """
        coeffs = np.array(
            [
                [0, 0, 1],
                [self.affine_width, 0, 1],
                [0, self.affine_height, 1],
            ],
            dtype=float,
        )

        for t in self.transforms:
            x_vals = np.array([t.x1, t.x2, t.x3], dtype=float)
            sol = np.linalg.solve(coeffs, x_vals)
            t.m[0, 0], t.m[1, 0], t.m[2, 0] = sol

            y_vals = np.array([t.y1, t.y2, t.y3], dtype=float)
            sol = np.linalg.solve(coeffs, y_vals)
            t.m[0, 1], t.m[1, 1], t.m[2, 1] = sol

            # Homogeneous row — always (0, 0, 1)
            t.m[0, 2] = 0.0
            t.m[1, 2] = 0.0
            t.m[2, 2] = 1.0

    def transform_point(self, x: float, y: float) -> tuple[float, float, object]:
        """Apply one randomly selected affine transform to (x, y).

        The transform is chosen by comparing a uniform random number against
        the stored CDF values (the chaos game algorithm).

        Parameters
        ----------
        x, y : float
            Current attractor point in world coordinates.

        Returns
        -------
        xt, yt : float
            Transformed point.
        color : str or pygame.Color
            Color associated with the selected transform.

        Notes
        -----
        Returns (0, 0, 0) as a sentinel if no transform is selected, which
        can happen when the probability weights do not sum to 1.0.
        """
        p = self._rng.random()
        for t in self.transforms:
            if p < t.probability:
                xt = x * t.m[0, 0] + y * t.m[1, 0] + t.m[2, 0]
                yt = x * t.m[0, 1] + y * t.m[1, 1] + t.m[2, 1]
                return xt, yt, t.color
        # Probabilities did not sum to 1.0 — return a safe sentinel
        return 0.0, 0.0, 0


def main():
    print("This module is intended to be imported, not executed directly.")


if __name__ == "__main__":
    main()
