#!/usr/bin/env -S uv run
"""k_means.py"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle

K_CLUSTERS = 3
INCLUDE_OUTLIER = False
MEAN_MULTIPLE = 0


@dataclass
class DataPoint:
    x = 0.0
    y = 0.0
    cluster = None


@dataclass
class Cluster:
    index = 0
    color = ""
    x = 0.0
    y = 0.0
    population = 0
    mean_distance = 0.0


def init_points():
    pts: list[DataPoint] = []
    samples = np.genfromtxt("cluster_samples.csv", delimiter=",")
    for s in samples:
        p = DataPoint()
        p.x = s[0]
        p.y = s[1]
        pts.append(p)
    if not INCLUDE_OUTLIER:
        pts.pop()
    return pts


def init_clusters():
    cs: list[Cluster] = []
    colors = ("red", "blue", "green", "purple", "yellow", "orange")
    for i in range(K_CLUSTERS):
        c = Cluster()
        c.index = i
        c.color = colors[i]
        cs.append(c)
    return cs


def init_assign(pts, cs):
    for i, p in enumerate(pts):
        p.cluster = cs[i % K_CLUSTERS]
        p.cluster.population += 1


def reassign(pts, cs):
    # Phase I: Calculate the new geometric mean of each
    # cluster based upon current data point assignments
    converged = True
    for c in cs:
        nx, ny = 0.0, 0.0
        for p in pts:
            if p.cluster.index == c.index:
                nx += p.x
                ny += p.y
        nx /= c.population
        ny /= c.population
        if c.x != nx or c.y != ny:
            c.x, c.y = nx, ny
            converged = False

    # Phase II: Assign data points to nearest cluster
    for p in pts:
        min_d = np.finfo(np.float64).max
        min_i = 0
        for c in cs:
            d = np.hypot(p.x - c.x, p.y - c.y)
            if d < min_d:
                min_d = d
                min_i = c.index
        if p.cluster.index != min_i and p.cluster.population > 1:
            p.cluster.population -= 1
            p.cluster = cs[min_i]
            p.cluster.population += 1
            converged = False

    # Phase III - Evict any point too far away from its cluster's center
    if converged and MEAN_MULTIPLE > 0:
        # Calculate mean distance from each cluster's center
        # to the assigned points for that cluster
        for c in cs:
            d = 0.0
            for p in pts:
                if p.cluster.index == c.index:
                    d += np.hypot(p.x - c.x, p.y - c.y)
            c.mean_distance = d / c.population

        # Only keep points where the distance to its assigned cluster's
        # center is less than a multiple of that cluster's mean distance
        # to its assigned points
        new_pts: list[DataPoint] = []
        for p in pts:
            c = p.cluster
            d = np.hypot(p.x - c.x, p.y - c.y)
            if d < c.mean_distance * MEAN_MULTIPLE:
                new_pts.append(p)
            else:
                if c.population > 1:
                    print(f"Evicted DataPoint({p.x}, {p.y}) from Cluster {c.index}")
                    c.population -= 1
                    converged = False
        pts[:] = new_pts

    return converged


def plot(pts, cs):
    for p in pts:
        plt.scatter(p.x, p.y, color=p.cluster.color, alpha=0.5, edgecolor="black")
    for c in cs:
        plt.scatter(c.x, c.y, color=c.color, marker=MarkerStyle("s"))
    plt.title(f"k-Means Clustering (k={K_CLUSTERS})")
    plt.xlim(-5, 45)
    plt.ylim(-5, 45)
    plt.gca().set_aspect("equal")


def on_key_press(event, pts, cs):
    if event.key == "n":
        if reassign(pts, cs):
            print("Cluster has converged!")
        plt.gca().clear()
        plot(pts, cs)
        plt.draw()


def main():
    points = init_points()
    clusters = init_clusters()
    init_assign(points, clusters)

    plt.figure(Path(__file__).name)
    plot(points, clusters)
    plt.gcf().canvas.mpl_connect(
        "key_press_event", lambda event: on_key_press(event, points, clusters)
    )
    plt.show()


if __name__ == "__main__":
    main()
