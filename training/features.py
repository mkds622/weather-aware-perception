import numpy as np


def safe_hist(x, bins, r):
    # histogram of values in range r split into bins
    h, _ = np.histogram(x, bins=bins, range=r)
    h = h.astype(np.float32)
    if h.sum() > 0:
        h /= h.sum()  # normalize to distribution
    return h


# -------- LIDAR --------
def lidar_features(path):
    # load raw lidar (x, y, z, intensity)
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    if len(pts) == 0:
        return np.zeros(45, dtype=np.float32)

    x, y, z, intensity = pts.T
    d = np.sqrt(x**2 + y**2 + z**2)  # distance

    f = []

    f.append(safe_hist(d, 10, (0, 100)))      # range distribution
    f.append(safe_hist(z, 10, (-5, 5)))       # height distribution

    # spatial density in XY plane (coarse grid)
    grid, _, _ = np.histogram2d(x, y, bins=5, range=[[-50, 50], [-50, 50]])
    grid = grid.flatten()
    if grid.sum() > 0:
        grid /= grid.sum()
    f.append(grid.astype(np.float32))

    # basic stats
    f.append(np.array([
        len(pts),
        np.mean(intensity),
        np.std(intensity),
        np.mean(d),
        np.std(d)
    ], dtype=np.float32))

    return np.concatenate(f)


# -------- RADAR --------
def radar_features(path):
    # load dict saved earlier (velocity, azimuth, altitude, depth)
    data = np.load(path, allow_pickle=True).item()

    d = data["depth"]      # distance
    v = data["velocity"]   # radial velocity

    if len(d) == 0:
        return np.zeros(25, dtype=np.float32)

    f = []

    f.append(safe_hist(d, 10, (0, 100)))      # range distribution
    f.append(safe_hist(v, 10, (-20, 20)))     # velocity distribution

    f.append(np.array([
        len(d),
        np.mean(v),
        np.std(v)
    ], dtype=np.float32))

    return np.concatenate(f)