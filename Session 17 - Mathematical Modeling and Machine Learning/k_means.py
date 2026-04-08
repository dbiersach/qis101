#!/usr/bin/env -S uv run
"""k_means.py - Interactive k-Means clustering with step-by-step convergence.

Press 'n' in the plot window to advance one iteration.  The algorithm runs
until convergence is detected and prints a message when it stabilizes.

Global knobs
------------
K_CLUSTERS : int
    Number of cluster centroids to fit.
INCLUDE_OUTLIER : bool
    When False the last row of cluster_samples.csv (an intentional outlier)
    is dropped before clustering begins.
MEAN_MULTIPLE : float
    If > 0, points farther than ``MEAN_MULTIPLE x cluster_mean_distance``
    from their centroid are evicted after convergence.  Set to 0 to disable.
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle

K_CLUSTERS: int = 3
INCLUDE_OUTLIER: bool = False
MEAN_MULTIPLE: float = 0


@dataclass
class DataPoint:
    """A single 2-D sample with its current cluster assignment.

    Parameters
    ----------
    x : float
        Horizontal coordinate.
    y : float
        Vertical coordinate.
    cluster : Cluster | None
        The cluster this point is currently assigned to.
    """

    x: float = field(default=0.0)
    y: float = field(default=0.0)
    cluster: "Cluster | None" = field(default=None)


@dataclass
class Cluster:
    """A cluster centroid and its bookkeeping metadata.

    Parameters
    ----------
    index : int
        Zero-based identifier used to cross-reference with DataPoint.cluster.
    color : str
        Matplotlib colour string used for all scatter plots.
    x : float
        Current centroid x-coordinate.
    y : float
        Current centroid y-coordinate.
    population : int
        Number of DataPoints currently assigned to this cluster.
    mean_distance : float
        Mean Euclidean distance from the centroid to its assigned points,
        computed only when MEAN_MULTIPLE > 0.
    """

    index: int = field(default=0)
    color: str = field(default="")
    x: float = field(default=0.0)
    y: float = field(default=0.0)
    population: int = field(default=0)
    mean_distance: float = field(default=0.0)


def init_points() -> list[DataPoint]:
    """Load data points from *cluster_samples.csv*.

    Returns
    -------
    list[DataPoint]
        All samples from the CSV, optionally with the trailing outlier removed
        according to the ``INCLUDE_OUTLIER`` flag.
    """
    samples = np.genfromtxt(
        Path(__file__).parent / "cluster_samples.csv", delimiter=","
    )
    pts: list[DataPoint] = [DataPoint(x=float(s[0]), y=float(s[1])) for s in samples]
    if not INCLUDE_OUTLIER:
        pts.pop()
    return pts


def init_clusters() -> list[Cluster]:
    """Create K_CLUSTERS empty Cluster objects with distinct colors.

    Returns
    -------
    list[Cluster]
        Freshly initialized clusters; centroids start at the origin.
    """
    colors = ("red", "blue", "green", "purple", "yellow", "orange")
    return [Cluster(index=i, color=colors[i]) for i in range(K_CLUSTERS)]


def init_assign(pts: list[DataPoint], cs: list[Cluster]) -> None:
    """Distribute points across clusters in round-robin order.

    This gives each cluster a non-empty starting population so that the
    first centroid calculation in ``reassign()`` is well-defined.

    Parameters
    ----------
    pts : list[DataPoint]
        All data points (mutated in-place).
    cs : list[Cluster]
        All clusters (mutated in-place).
    """
    for i, p in enumerate(pts):
        p.cluster = cs[i % K_CLUSTERS]
        p.cluster.population += 1


def reassign(pts: list[DataPoint], cs: list[Cluster]) -> bool:
    """Advance the k-Means algorithm by one full iteration.

    Three phases are executed in sequence:

    Phase I — Recompute centroids
        Each centroid moves to the geometric mean of its currently assigned
        points.  NumPy array operations replace the previous nested loop,
        dropping the per-cluster cost from O(N) with Python overhead to a
        single vectorized pass.

    Phase II — Reassign points
        Every point is reassigned to its nearest centroid.  Only moves that
        keep all cluster populations ≥ 1 are applied.

    Phase III — Evict outliers  (only when converged and MEAN_MULTIPLE > 0)
        Points whose distance to their centroid exceeds
        ``MEAN_MULTIPLE × cluster_mean_distance`` are removed entirely.

    Parameters
    ----------
    pts : list[DataPoint]
        All active data points.  **Mutated in-place**; Phase III may shrink
        the list by removing evicted outliers.
    cs : list[Cluster]
        All clusters.  **Mutated in-place** (centroids and populations).

    Returns
    -------
    bool
        ``True`` when no centroid moved and no point changed cluster
        (i.e. the algorithm has converged).
    """
    converged = True

    # ------------------------------------------------------------------
    # Phase I: recompute each centroid using vectorized NumPy means
    # ------------------------------------------------------------------
    xs = np.array([p.x for p in pts])
    ys = np.array([p.y for p in pts])
    labels = np.array([p.cluster.index for p in pts])

    for c in cs:
        mask = labels == c.index
        if not mask.any():
            continue
        nx = float(xs[mask].mean())
        ny = float(ys[mask].mean())
        if c.x != nx or c.y != ny:
            c.x, c.y = nx, ny
            converged = False

    # ------------------------------------------------------------------
    # Phase II: reassign each point to its nearest centroid
    # ------------------------------------------------------------------
    centroids_x = np.array([c.x for c in cs])
    centroids_y = np.array([c.y for c in cs])

    for p in pts:
        distances = np.hypot(p.x - centroids_x, p.y - centroids_y)
        min_i = int(distances.argmin())
        if p.cluster.index != min_i and p.cluster.population > 1:
            p.cluster.population -= 1
            p.cluster = cs[min_i]
            p.cluster.population += 1
            converged = False

    # ------------------------------------------------------------------
    # Phase III: evict outliers once converged (optional)
    # ------------------------------------------------------------------
    if converged and MEAN_MULTIPLE > 0:
        # Recompute per-cluster mean distance from centroid to members
        for c in cs:
            member_xs = np.array([p.x for p in pts if p.cluster.index == c.index])
            member_ys = np.array([p.y for p in pts if p.cluster.index == c.index])
            c.mean_distance = float(np.hypot(member_xs - c.x, member_ys - c.y).mean())

        # Retain only points within the distance threshold
        surviving: list[DataPoint] = []
        for p in pts:
            c = p.cluster
            d = np.hypot(p.x - c.x, p.y - c.y)
            if d < c.mean_distance * MEAN_MULTIPLE:
                surviving.append(p)
            elif c.population > 1:
                print(f"Evicted DataPoint({p.x}, {p.y}) from Cluster {c.index}")
                c.population -= 1
                converged = False

        pts[:] = surviving

    return converged


def plot(pts: list[DataPoint], cs: list[Cluster]) -> None:
    """Render all data points and centroids onto the current axes.

    Parameters
    ----------
    pts : list[DataPoint]
        Points to scatter; each is coloured by its cluster assignment.
    cs : list[Cluster]
        Centroids drawn as filled squares.
    """
    for p in pts:
        plt.scatter(p.x, p.y, color=p.cluster.color, alpha=0.5, edgecolor="black")
    for c in cs:
        plt.scatter(c.x, c.y, color=c.color, marker=MarkerStyle("s"))
    plt.title(f"k-Means Clustering (k={K_CLUSTERS})")
    plt.xlim(-5, 45)
    plt.ylim(-5, 45)
    plt.gca().set_aspect("equal")


def on_key_press(event, pts: list[DataPoint], cs: list[Cluster]) -> None:
    """Handle keyboard events for the interactive plot.

    Press **n** to step the algorithm forward by one iteration.

    Parameters
    ----------
    event : matplotlib.backend_bases.KeyEvent
        The key-press event forwarded by matplotlib.
    pts : list[DataPoint]
        Shared list of data points passed through to ``reassign``.
    cs : list[Cluster]
        Shared list of clusters passed through to ``reassign``.
    """
    if event.key == "n":
        if reassign(pts, cs):
            print("Cluster has converged!")
        plt.gca().clear()
        plot(pts, cs)
        plt.draw()


def main() -> None:
    """Entry point: load data, initialise clusters, and open the plot window."""
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
