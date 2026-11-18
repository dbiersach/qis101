#!/usr/bin/env -S uv run
"""euler_line.py"""

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

prng_seed = 2018


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class RadLine:
    theta: float
    d: float


# Initialize global variables
ptA = Point2D(x=0.0, y=0.0)
ptB = Point2D(x=0.0, y=0.0)
ptC = Point2D(x=0.0, y=0.0)
ptOrtho = Point2D(x=0.0, y=0.0)
ptCircum = Point2D(x=0.0, y=0.0)

rlineAB = RadLine(theta=0.0, d=0.0)
rlineAC = RadLine(theta=0.0, d=0.0)
rlineBC = RadLine(theta=0.0, d=0.0)


def plot_radline(ax, rline, t, **kwargs):
    # Parameterize along the line instead of solving for y as a function of x.
    c, s = np.cos(rline.theta), np.sin(rline.theta)
    ax.plot(rline.d * c - t * s, rline.d * s + t * c, **kwargs)


def radline_connect(pt1, pt2):
    if pt1 == pt2:
        raise ValueError("A line needs two distinct points")
    theta = np.arctan2(pt2.x - pt1.x, pt1.y - pt2.y) % np.pi
    d = pt1.x * np.cos(theta) + pt1.y * np.sin(theta)
    return RadLine(theta, d)


def radline_tangent(rline, pt):
    theta = rline.theta
    d = rline.d
    # Rotate radon line by 90 degrees, while ensuring
    # theta remains within the interval [0, pi)
    if theta < np.pi / 2:
        theta += np.pi / 2
    else:
        theta -= np.pi / 2
    d = pt.x * np.cos(theta) + pt.y * np.sin(theta)
    return RadLine(theta, d)


def radline_intersect(rline1, rline2):
    theta1 = rline1.theta
    d1 = rline1.d
    theta2 = rline2.theta
    d2 = rline2.d
    # If two radon lines have the same theta, then
    # they are parallel and can never intersect
    if theta1 == theta2:
        return Point2D(np.nan, np.nan)
    z = float(np.sin(theta2 - theta1))
    x = float((d1 * np.sin(theta2) - d2 * np.sin(theta1)) / z)
    y = float((d2 * np.cos(theta1) - d1 * np.cos(theta2)) / z)
    return Point2D(x, y)


def random_pt():
    x = random.randint(-10, 10)
    y = random.randint(-10, 10)
    return Point2D(x, y)


def plot_triangle(ax):
    # Generate three random vertices
    global ptA, ptB, ptC
    while True:
        ptA, ptB, ptC = random_pt(), random_pt(), random_pt()
        twice_area = ((ptB.x - ptA.x) * (ptC.y - ptA.y)
                      - (ptB.y - ptA.y) * (ptC.x - ptA.x))
        if twice_area != 0:
            break

    # Formulate the radon line connecting the adjacent vertices
    global rlineAB, rlineAC, rlineBC
    rlineAB = radline_connect(ptA, ptB)
    rlineAC = radline_connect(ptA, ptC)
    rlineBC = radline_connect(ptB, ptC)

    # Draw each edge directly between its vertices, including vertical edges.
    ax.plot([ptA.x, ptB.x], [ptA.y, ptB.y], color="purple", linewidth=3, label="Edges")
    ax.plot([ptA.x, ptC.x], [ptA.y, ptC.y], color="purple", linewidth=3)
    ax.plot([ptB.x, ptC.x], [ptB.y, ptC.y], color="purple", linewidth=3)

    # Plot the extended edges
    x = np.linspace(-100, 100, 100)
    plot_radline(ax, rlineAB, x, color="purple", linestyle="dotted")
    plot_radline(ax, rlineAC, x, color="purple", linestyle="dotted")
    plot_radline(ax, rlineBC, x, color="purple", linestyle="dotted")

    # Calculate the centroid, which is the mean value
    # of the cartesian coordinates of every vertex
    ptCentroid = Point2D(
        x=(ptA.x + ptB.x + ptC.x) / 3.0, y=(ptA.y + ptB.y + ptC.y) / 3.0
    )

    # Plot the centroid point
    ax.scatter(ptCentroid.x, ptCentroid.y, color="purple", s=100, label="Centroid")


def plot_orthocenter(ax):
    # Formulate the radon lines describing each altitude, which is a line
    # extending from each vertex that is perpendicular to the opposite edge
    rlineA_alt = radline_tangent(rlineBC, ptA)
    rlineB_alt = radline_tangent(rlineAC, ptB)
    rlineC_alt = radline_tangent(rlineAB, ptC)

    # Plot the altitudes
    x = np.linspace(-100, 100, 100)
    plot_radline(ax, rlineA_alt, x, color="blue", linestyle="dotted", label="Altitudes"
    )
    plot_radline(ax, rlineB_alt, x, color="blue", linestyle="dotted")
    plot_radline(ax, rlineC_alt, x, color="blue", linestyle="dotted")

    # Calculate the orthocenter which is the intersection of the altitudes
    global ptOrtho
    ptOrtho = radline_intersect(rlineA_alt, rlineB_alt)

    # Plot the orthocenter point
    ax.scatter(ptOrtho.x, ptOrtho.y, color="blue", s=100, label="Orthocenter")


def plot_circumcenter(ax):
    # Calculate the midpoints of each edge
    ptAB_mid = Point2D(x=(ptA.x + ptB.x) / 2.0, y=(ptA.y + ptB.y) / 2.0)
    ptAC_mid = Point2D(x=(ptA.x + ptC.x) / 2.0, y=(ptA.y + ptC.y) / 2.0)
    ptBC_mid = Point2D(x=(ptB.x + ptC.x) / 2.0, y=(ptB.y + ptC.y) / 2.0)

    # Calculate the radon line that is the
    # perpendicular bisector of each edge
    rlineA_bis = radline_tangent(rlineBC, ptBC_mid)
    rlineB_bis = radline_tangent(rlineAC, ptAC_mid)
    rlineC_bis = radline_tangent(rlineAB, ptAB_mid)

    # Plot the perpendicular bisectors
    x = np.linspace(-100, 100, 100)
    plot_radline(ax, rlineA_bis, x,
        color="green",
        linestyle="dotted",
        label="Perpendicular Bisectors",
    )
    plot_radline(ax, rlineB_bis, x, color="green", linestyle="dotted")
    plot_radline(ax, rlineC_bis, x, color="green", linestyle="dotted")

    # Calculate the circumcenter, which is the intersection
    # of the perpendicular bisectors for each edge
    global ptCircum
    ptCircum = radline_intersect(rlineA_bis, rlineB_bis)

    # Plot the circumcenter point
    ax.scatter(ptCircum.x, ptCircum.y, color="green", s=100, label="Circumcenter")


def plot_euler_Line(ax):
    # Plot the Euler line, which intersects the orthocenter,
    # the circumcenter, and the centroid of every triangle
    x = np.linspace(-100, 100, 100)
    if np.allclose((ptOrtho.x, ptOrtho.y), (ptCircum.x, ptCircum.y)):
        return
    rlineEuler = radline_connect(ptOrtho, ptCircum)
    plot_radline(ax, rlineEuler, x, color="black", linewidth=3, label="Euler line")


def plot_incenter(ax):
    # Find the length of each edge
    dAB = np.sqrt((ptA.x - ptB.x) ** 2 + (ptA.y - ptB.y) ** 2)
    dAC = np.sqrt((ptA.x - ptC.x) ** 2 + (ptA.y - ptC.y) ** 2)
    dBC = np.sqrt((ptB.x - ptC.x) ** 2 + (ptB.y - ptC.y) ** 2)

    # Assume the angle bisectors are the sample mean of each adjacent edge
    theta_bis_a = (rlineAB.theta + rlineAC.theta) / 2.0
    d_bis_a = ptA.x * np.cos(theta_bis_a) + ptA.y * np.sin(theta_bis_a)
    rlineA_inctr = RadLine(theta_bis_a, d_bis_a)

    theta_bis_b = (rlineAB.theta + rlineBC.theta) / 2.0
    d_bis_b = ptB.x * np.cos(theta_bis_b) + ptB.y * np.sin(theta_bis_b)
    rlineB_inctr = RadLine(theta_bis_b, d_bis_b)

    theta_bis_c = (rlineAC.theta + rlineBC.theta) / 2.0
    d_bis_c = ptC.x * np.cos(theta_bis_c) + ptC.y * np.sin(theta_bis_c)
    rlineC_inctr = RadLine(theta_bis_c, d_bis_c)

    # Find the points where the angle bisectors intersect
    ptAB = radline_intersect(rlineA_inctr, rlineB_inctr)
    ptAC = radline_intersect(rlineA_inctr, rlineC_inctr)
    ptBC = radline_intersect(rlineB_inctr, rlineC_inctr)

    # Find the distance between each vertex and the intersection point of
    # its own angle bisector and the angle bisector of its adjacent vertices
    dA_AB = np.sqrt((ptA.x - ptAB.x) ** 2 + (ptA.y - ptAB.y) ** 2)
    dB_AB = np.sqrt((ptB.x - ptAB.x) ** 2 + (ptB.y - ptAB.y) ** 2)
    dA_AC = np.sqrt((ptA.x - ptAC.x) ** 2 + (ptA.y - ptAC.y) ** 2)
    dC_AC = np.sqrt((ptC.x - ptAC.x) ** 2 + (ptC.y - ptAC.y) ** 2)
    dB_BC = np.sqrt((ptB.x - ptBC.x) ** 2 + (ptB.y - ptBC.y) ** 2)
    dC_BC = np.sqrt((ptC.x - ptBC.x) ** 2 + (ptC.y - ptBC.y) ** 2)

    # Rotate by 90 degrees any bisector which results in bisector intersection
    # points that fall outside the triangle, as the incenter must be in the triangle
    # If a rotation is necessary, recalculate the Radon Line for that bisector
    if dB_BC <= dBC and dC_BC <= dBC:
        theta_bis_a -= np.pi / 2
        d_bis_a = ptA.x * np.cos(theta_bis_a) + ptA.y * np.sin(theta_bis_a)
        rlineA_inctr = RadLine(theta_bis_a, d_bis_a)
    elif dA_AC <= dAC and dC_AC < dAC:
        theta_bis_b -= np.pi / 2
        d_bis_b = ptB.x * np.cos(theta_bis_b) + ptB.y * np.sin(theta_bis_b)
        rlineB_inctr = RadLine(theta_bis_b, d_bis_b)
    elif dA_AB < dAB and dB_AB < dAB:
        theta_bis_c -= np.pi / 2
        d_bis_c = ptC.x * np.cos(theta_bis_c) + ptC.y * np.sin(theta_bis_c)
        rlineC_inctr = RadLine(theta_bis_c, d_bis_c)

    # Plot the angle bisectors of each vertex
    x = np.linspace(-100, 100, 100)
    plot_radline(ax, rlineA_inctr, x,
        color="red",
        label="Angle Bisectors",
        linestyle="dotted",
    )
    plot_radline(ax, rlineB_inctr, x, color="red", linestyle="dotted")
    plot_radline(ax, rlineC_inctr, x, color="red", linestyle="dotted")

    # Plot the incenter, which is the intersection of the
    # angle bisectors of every vertex
    ptIncenter = radline_intersect(rlineA_inctr, rlineB_inctr)
    ax.scatter(ptIncenter.x, ptIncenter.y, color="red", s=100, label="Incenter")


def main():
    random.seed(prng_seed)

    plt.figure(Path(__file__).name, constrained_layout=True)
    plt.get_current_fig_manager().full_screen_toggle()
    ax = plt.gca()

    plot_triangle(ax)
    plot_circumcenter(ax)
    plot_orthocenter(ax)
    plot_euler_Line(ax)

    # Note: the triangle incenter is falls on the Euler Line
    # if and only if the triangle is isosceles
    plot_incenter(ax)

    plt.title(f"Euler Line (seed={prng_seed})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.legend(loc="upper right")
    ax.grid("on")
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
