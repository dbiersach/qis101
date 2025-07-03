#!/usr/bin/env python3
"""maze_draw.py"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle


def validate_maze(maze):
    for y in range(10):
        for x in range(10):
            cell = int(maze[y, x])
            # Check CSV has a value in every cell
            if cell == -1:
                print(f"Cell {y, x} has no value")
                return False
            # Check for any holes in border walls of maze
            if y == 0 and cell & 1 != 1:
                print(f"Cell {y, x} is missing the NORTH wall")
                return False
            if x == 9 and cell & 2 != 2:
                print(f"Cell {y, x} is missing the EAST wall")
                return False
            if y == 9 and cell & 4 != 4:
                print(f"Cell {y, x} is missing the SOUTH wall")
                return False
            if x == 0 and cell & 8 != 8:
                print(f"Cell {y, x} is missing the WEST wall")
                return False
            # Check every cell agrees with its NORTH cell
            if y > 0:
                cell2: int = int(maze[y - 1, x])
                if (cell & 1 == 1 and cell2 & 4 != 4) or (
                    cell & 1 != 1 and cell2 & 4 == 4
                ):
                    print(
                        (
                            f"Cell {y, x}={cell} and cell {y - 1, x}={cell2}"
                            " do not agree between NORTH/SOUTH"
                        )
                    )
                    return False
            # Check every cell agrees with its SOUTH cell
            if y < 9:
                cell2 = int(maze[y + 1, x])
                if (cell & 4 == 4 and cell2 & 1 != 1) or (
                    cell & 4 != 4 and cell2 & 1 == 1
                ):
                    print(
                        (
                            f"Cell {y, x}={cell} and cell {y + 1, x}={cell2}"
                            " do not agree between NORTH/SOUTH"
                        )
                    )
                    return False
            # Check every cell agrees with its EAST cell
            if x < 9:
                cell2 = int(maze[y, x + 1])
                if (cell & 2 == 2 and cell2 & 8 != 8) or (
                    cell & 2 != 2 and cell2 & 8 == 8
                ):
                    print(
                        (
                            f"Cell {y, x}={cell} and cell {y, x + 1}={cell2}"
                            " do not agree between EAST/WEST"
                        )
                    )
                    return False
            # Check every cell agrees with its WEST cell
            if x > 0:
                cell2 = int(maze[y, x - 1])
                if (cell & 8 == 8 and cell2 & 2 != 2) or (
                    cell & 8 != 8 and cell2 & 2 == 2
                ):
                    print(
                        (
                            f"Cell {y, x}={cell} and cell {y, x - 1}={cell2}"
                            " do not agree between EAST/WEST"
                        )
                    )
                    return False
    return True


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


def plot_maze(ax, maze):
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

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


def main(file_name):
    file_path = Path(__file__).parent / file_name
    m = np.genfromtxt(file_path, delimiter=",", dtype=int)
    if validate_maze(m):
        file_path = file_path.with_suffix(".pickle")
        with open(file_path, "wb") as outfile:
            pickle.dump(m, outfile, pickle.HIGHEST_PROTOCOL)
        plt.figure(f"{Path(__file__).name} ({file_name})")
        plot_maze(plt.gca(), m)
        plt.show()


if __name__ == "__main__":
    main("maze.csv")
