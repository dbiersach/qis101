#!/usr/bin/env python3
"""convex_hull.py"""

from pathlib import Path

import matplotlib.pyplot as plt


def orientation(p, q, r):
    # Vector A points from P towards Q
    ax = q[0] - p[0]
    ay = q[1] - p[1]

    # Vector B points from Q towards R
    bx = r[0] - q[0]
    by = r[1] - q[1]

    # Calculate cross product between A and B
    val = (ax * by) - (ay * bx)

    if val == 0:
        # Points P, Q, and R are colinear
        return 0

    if val < 0:
        # Point R is to the left of Q
        # as seen from point P
        return -1

    if val > 0:
        # Point R is to the right of Q
        # as seen from point P
        return 1


def convex_hull(points):
    hull = []
    start = min(points)
    p = start

    while True:
        hull.append(p)
        q = points[0]

        for r in points:
            if r == p:
                continue
            turn = orientation(p, q, r)
            if (
                q == p
                or turn == -1
                or (turn == 0 and (r[0] > q[0] or (r[0] == q[0] and r[1] > q[1])))
            ):
                q = r

        p = q

        if p == start:
            break

    return hull


def main():
    # Declare 2D Cartesian points as a list of tuples
    # fmt: off
    points = [(1.5, 3), (2, 3), (2.5, 1.5),
            (2.5, 2.5), (2, 2), (3, 1),
            (1.5, 2.5), (1.5, 1.5), (1, 3.5)]
    # fmt: on

    # Calculate list of coordinates of the convex hull
    # using the Jarvis March algorithm
    hull = convex_hull(points)

    # Split the points into separate lists of the x and y coordinates
    h_x, h_y = zip(*hull)

    # Close the hull by connecting back to the first point
    h_x = list(h_x) + [h_x[0]]
    h_y = list(h_y) + [h_y[0]]

    # Split the points into separate lists of the x and y coordinates
    pt_x, pt_y = zip(*points)

    # Plot the points and the convex hull
    plt.figure(Path(__file__).name)
    plt.scatter(pt_x, pt_y, color="blue", zorder=2)
    plt.plot(h_x, h_y, color="red")
    plt.title("Convex Hull (Jarvis March)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.grid("on")
    plt.show()


if __name__ == "__main__":
    main()
