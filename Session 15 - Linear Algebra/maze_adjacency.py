#!/usr/bin/env -S uv run
"""maze_adjacency.py"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

total_steps = 0


def link_cells(adj, y1, x1, y2, x2):
    cell1: int = y1 * 10 + x1
    cell2: int = y2 * 10 + x2
    adj[cell1, cell2] = True
    adj[cell2, cell1] = True


def init_adjmat(maze):
    adj = np.zeros(10000, dtype=np.bool_).reshape((100, 100))
    for y in range(10):
        for x in range(10):
            cell: int = maze[y, x]
            if y > 0 and cell & 1 != 1:
                link_cells(adj, y, x, y - 1, x)
            if y < 9 and cell & 4 != 4:
                link_cells(adj, y, x, y + 1, x)
            if x < 9 and cell & 2 != 2:
                link_cells(adj, y, x, y, x + 1)
            if x > 0 and cell & 8 != 8:
                link_cells(adj, y, x, y, x - 1)

    return adj


def max_steps(adj_matrix):
    adj = np.copy(adj_matrix)
    steps = 0
    # Keep multiplying until the entrance and exit are "linked"
    while not (adj[0, 99] & adj[99, 0]):
        # Use @ operator for matrix multiplication
        adj = adj @ adj_matrix
        steps += 1
    # Add +2 to max steps, because we need +1 to step into
    # the entrance cell, and another +1 to enter the exit cell
    steps += 2
    return steps


def plot_cell_walls(ax, maze):
    for y in range(10):
        bottom: int = (9 - y) * 10
        top: int = bottom + 10
        for x in range(10):
            left: int = x * 10
            right: int = left + 10
            cell: int = maze[y, x]
            if cell & 1 == 1:
                lc = LineCollection(
                    [[(left, top), (right, top)]], color="black", linewidth=3
                )
                ax.add_collection(lc)
            if cell & 2 == 2:
                lc = LineCollection(
                    [[(right, bottom), (right, top)]], color="black", linewidth=3
                )
                ax.add_collection(lc)
            if cell & 4 == 4:
                lc = LineCollection(
                    [[(left, bottom), (right, bottom)]], color="black", linewidth=3
                )
                ax.add_collection(lc)
            if cell & 8 == 8:
                lc = LineCollection(
                    [[(left, bottom), (left, top)]], color="black", linewidth=3
                )
                ax.add_collection(lc)


def plot_steps(ax, steps):
    for step in steps:
        y, x, _ = step
        bottom: int = (9 - y) * 10
        left: int = x * 10
        patch = Rectangle((left + 4, bottom + 4), 2, 2)
        ax.add_collection(PatchCollection([patch], facecolor="blue"))


def plot_maze(ax, maze, steps):
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title(f"{len(steps)} steps ({total_steps} total)")

    # Plot enter and exit cells
    entrance = Rectangle((0, 90), 10, 10)
    ax.add_collection(PatchCollection([entrance], facecolor="tan"))
    exit = Rectangle((90, 0), 10, 10)
    ax.add_collection(PatchCollection([exit], facecolor="orange"))

    # Plot cell corner circles
    for x in range(0, 110, 10):
        for y in range(0, 110, 10):
            ax.scatter(x, y, color="black")

    plot_cell_walls(ax, maze)
    plot_steps(ax, steps)


def in_path(steps, y, x):
    for step in reversed(steps):
        sy, sx, _ = step
        if sy == y and sx == x:
            return True
    return False


def search_maze(maze, steps, step_limit):
    global total_steps

    y, x, dir = steps.pop()
    total_steps = total_steps + 1

    if x == 9 and y == 9:
        steps.append((9, 9, 0))
        return True

    if len(steps) == step_limit:
        return False

    if dir == 0:
        steps.append((y, x, 1))
        if maze[y, x] & 1 != 1 and not in_path(steps, y - 1, x):
            steps.append((y - 1, x, 0))
        return False

    if dir == 1:
        steps.append((y, x, 2))
        if maze[y, x] & 2 != 2 and not in_path(steps, y, x + 1):
            steps.append((y, x + 1, 0))
        return False

    if dir == 2:
        steps.append((y, x, 4))
        if maze[y, x] & 4 != 4 and not in_path(steps, y + 1, x):
            steps.append((y + 1, x, 0))
        return False

    if dir == 4:
        steps.append((y, x, 8))
        if maze[y, x] & 8 != 8 and not in_path(steps, y, x - 1):
            steps.append((y, x - 1, 0))
        return False

    return False


def main(file_name):
    file_path = Path(__file__).parent / file_name
    with open(file_path, "rb") as infile:
        m = pickle.load(infile)

    # Steps is list of tuples, where each tuple contains the
    # (y, x, last direction tried) for each step along current path
    s = [(0, 0, 0)]

    # Create boolean adjacency matrix from the wall data
    adj_mat = init_adjmat(m)

    # Calculate maximum number of allowed steps along shortest path
    limit = max_steps(adj_mat)

    while not search_maze(m, s, limit):
        pass

    plt.figure(f"{Path(__file__).name} ({file_name})")
    plot_maze(plt.gca(), m, s)
    plt.show()


if __name__ == "__main__":
    main("maze.pickle")
