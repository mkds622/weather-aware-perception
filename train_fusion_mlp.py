"""
Full fusion model: LiDAR + Radar + Camera

- Radar (4 sensors)
- LiDAR (.bin point cloud features)
- Camera features

"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import cv2
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = Path("raw/weather_dataset_fusion_extended_2")  # update if needed
MODEL_DIR = Path("models/carla/fusion_lrc")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REGIME_MAP = {"clear": 0, "rain": 1, "fog": 2}


# ---------- CAMERA ----------

def extract_camera_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.array([
        gray.mean(),
        gray.std(),
        gray.min(),
        gray.max(),
        np.percentile(gray, 25),
        np.percentile(gray, 50),
        np.percentile(gray, 75),
        np.mean(gray < 50),
        np.mean(gray > 200),
        cv2.Laplacian(gray, cv2.CV_64F).var()
    ])


# ---------- RADAR ----------

def extract_radar_features(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()

    d = data["depth"]
    az = data["azimuth"]

    if len(d) == 0:
        return None

    features = []

    features.append(len(d))
    features.extend([d.mean(), d.std(), d.min(), d.max()])

    bins = [-np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3]
    features.extend(np.histogram(az, bins=bins)[0])

    depth_bins = np.linspace(0, 100, 6)
    features.extend(np.histogram(d, bins=depth_bins)[0])

    features.append(np.mean(d > 50))
    features.append(np.mean(d < 10))
    features.append(np.std(az))
    features.append(np.mean(np.abs(np.diff(np.sort(az)))))

    features.append(len(d) / 9000)
    features.append(np.std(d))
    features.append(np.var(d))

    return np.array(features)  # 21


# ---------- LIDAR ----------

def extract_lidar_features(bin_path):
    pts = np.fromfile(bin_path, dtype=np.float32)

    if pts.shape[0] == 0:
        return None

    pts = pts.reshape(-1, 4)  # x, y, z, intensity

    d = np.linalg.norm(pts[:, :3], axis=1)

    return np.array([
        len(d),
        d.mean(),
        d.std(),
        d.min(),
        d.max(),
        np.percentile(d, 25),
        np.percentile(d, 50),
        np.percentile(d, 75),
        np.mean(d > 50),
        np.mean(d < 10)
    ])


# ---------- FRAME ----------

def extract_frame_features(regime_dir, fid):
    radar_front = regime_dir / "ego" / "radar_front"
    radar_back = regime_dir / "ego" / "radar_back"
    lidar_dir = regime_dir / "ego" / "lidar"
    cam_dir = regime_dir / "ego" / "camera_front"

    # radar (4 sensors)
    radar_paths = [
        radar_front / f"left_{fid}.npy",
        radar_front / f"right_{fid}.npy",
        radar_back / f"left_{fid}.npy",
        radar_back / f"right_{fid}.npy"
    ]

    radar_feats = []
    for p in radar_paths:
        if not p.exists():
            return None
        f = extract_radar_features(p)
        if f is None:
            return None
        radar_feats.append(f)

    radar_feats = np.concatenate(radar_feats)  # 84

    # lidar
    lidar_path = lidar_dir / f"{fid}.bin"
    if not lidar_path.exists():
        print("Missing lidar:", lidar_path)
        return None

    lidar_feats = extract_lidar_features(lidar_path)
    if lidar_feats is None:
        return None
    
    # camera
    cam_path = cam_dir / f"{fid}.png"
    if not cam_path.exists():
        return None
    cam_feats = extract_camera_features(cam_path)
    if cam_feats is None:
        return None

    return np.concatenate([radar_feats, lidar_feats, cam_feats])  
    # 84 + 10 + 10 = 104


# ---------- DATASET ----------

def build_dataset(run_dirs):
    X, y = [], []

    for run_dir in run_dirs:
        for regime in ["clear", "rain", "fog"]:
            regime_dir = run_dir / regime
            radar_front = regime_dir / "ego" / "radar_front"

            if not radar_front.exists():
                continue

            for f in radar_front.glob("left_*.npy"):
                fid = f.stem.split("_")[-1]

                feat = extract_frame_features(regime_dir, fid)
                if feat is None:
                    continue

                X.append(feat)
                y.append(REGIME_MAP[regime])

    return np.array(X), np.array(y)


# ---------- MODEL ----------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(104, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)


# ---------- TRAIN ----------

def train():
    runs = sorted(DATA_ROOT.glob("run_*"))
    random.shuffle(runs)

    split = int(0.8 * len(runs))
    train_runs = runs[:split]
    val_runs = runs[split:]

    X_train, y_train = build_dataset(train_runs)
    X_val, y_val = build_dataset(val_runs)

    print("Train:", X_train.shape, "Val:", X_val.shape)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(25):
        model.train()
        loss = loss_fn(model(X_train), y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X_val).argmax(1) == y_val).float().mean().item()

        print(f"Epoch {epoch:02d} | Acc {acc:.3f}")


if __name__ == "__main__":
    train()