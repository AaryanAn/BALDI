import numpy as np


def dtw_distance(a, b):
    """Simple DTW on 2D trajectories."""
    if a is None or b is None:
        return float("inf")

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim != 2 or b.ndim != 2:
        return float("inf")

    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            d = float(np.linalg.norm(ai - bj))
            best_prev = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            cost[i, j] = d + best_prev

    dist = float(cost[n, m])
    norm = float(n + m)
    if norm > 0.0:
        dist = dist / norm

    return dist

