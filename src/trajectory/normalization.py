import numpy as np


def flatten_paths(paths):
    """paths: list of strokes, each stroke a list of (x, y) points"""
    if not paths:
        return np.zeros((0, 2), dtype=np.float32)

    points = []
    for stroke in paths:
        for x, y in stroke:
            points.append([x, y])

    if not points:
        return np.zeros((0, 2), dtype=np.float32)

    return np.array(points, dtype=np.float32)


def resample(points, num_points=100):
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, num_points, axis=0)

    # parameterize by cumulative distance
    deltas = points[1:] - points[:-1]
    seg_lengths = np.linalg.norm(deltas, axis=1)
    total_length = float(seg_lengths.sum())

    if total_length == 0.0:
        return np.repeat(points[:1], num_points, axis=0)

    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    target = np.linspace(0.0, total_length, num_points)

    resampled = []
    j = 0
    for t in target:
        while j + 1 < len(cum_lengths) and cum_lengths[j + 1] < t:
            j += 1

        if j + 1 == len(cum_lengths):
            resampled.append(points[-1])
        else:
            t0 = cum_lengths[j]
            t1 = cum_lengths[j + 1]
            alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            p = (1.0 - alpha) * points[j] + alpha * points[j + 1]
            resampled.append(p)

    return np.stack(resampled).astype(np.float32)


def normalize(points):
    if len(points) == 0:
        return points

    pts = points.astype(np.float32)

    # center at origin
    mean = pts.mean(axis=0, keepdims=True)
    pts = pts - mean

    # scale to unit size
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0.0:
        pts = pts / max_norm

    # align main axis
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argmax(eigvals)
    main_axis = eigvecs[:, idx]

    # rotate so main axis roughly points to +x
    angle = np.arctan2(main_axis[1], main_axis[0])
    c = np.cos(-angle)
    s = np.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ rot.T

    return pts


def preprocess_paths(paths, num_points=100):
    """Takes raw paths from Gestures and returns a normalized trajectory."""
    pts = flatten_paths(paths)
    if len(pts) == 0:
        return pts

    pts = resample(pts, num_points=num_points)
    pts = normalize(pts)
    return pts

