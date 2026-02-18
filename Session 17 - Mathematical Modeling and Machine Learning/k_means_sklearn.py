#!/usr/bin/env -S uv run
"""k_means_sklearn.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

K_CLUSTERS = 3
INCLUDE_OUTLIER = False

file_name = "cluster_samples.csv"
file_path = Path(__file__).parent / file_name
points = np.genfromtxt(file_path, delimiter=",")
if not INCLUDE_OUTLIER:
    points = points[:-1]

kmeans = KMeans(K_CLUSTERS)
kmeans.fit(points)
clusters = kmeans.predict(points)
centers = kmeans.cluster_centers_

plt.figure(Path(__file__).name)
plt.title(f"k-Means Clustering (k={K_CLUSTERS})")
plt.scatter(points[:, 0], points[:, 1], c=clusters)
plt.scatter(centers[:, 0], centers[:, 1], color="black", facecolor="none", marker="s")
plt.xlim(-5, 45)
plt.ylim(-5, 45)
plt.gca().set_aspect("equal")
plt.show()
