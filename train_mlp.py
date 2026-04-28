"""
Fusion model: Camera + Radar (CARLA)

- Combines camera and radar features
- Demonstrates dominance of camera features

"""


from math import perm
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random
import cv2
from training import features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = Path("/home/meet/projects/RAWAP_carla_l4dr/weather-aware-perception/raw/weather_dataset_extended_2")
MODEL_DIR = Path("models/carla/fusion_rc_extended_2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REGIME_MAP = {"clear": 0, "fog": 1, "rain": 2}

# ---------- FEATURE EXTRACTION ----------

def extract_camera_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros(10)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.array([
        gray.mean(),
        gray.std(),
        gray.min(),
        gray.max(),
        np.percentile(gray, 25),
        np.percentile(gray, 50),
        np.percentile(gray, 75),
        cv2.Laplacian(gray, cv2.CV_64F).var(),  # blur
        np.mean(gray < 50),   # dark ratio
        np.mean(gray > 200)   # bright ratio
    ])

def extract_radar_features(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()

    d = data["depth"]
    az = data["azimuth"]

    if len(d) == 0:
        return np.zeros(21)

    features = []

    # 1. density
    features.append(len(d))

    # 2. distance stats
    features.extend([
        d.mean(),
        d.std(),
        d.min(),
        d.max()
    ])

    # 3. sector bins (azimuth)
    bins = [-np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3]
    sector_counts = np.histogram(az, bins=bins)[0]
    features.extend(sector_counts)

    # 4. depth histogram
    depth_bins = np.linspace(0, 100, 6)
    depth_hist = np.histogram(d, bins=depth_bins)[0]
    features.extend(depth_hist)

    features.append(np.mean(d > 50))   # far return ratio
    features.append(np.mean(d < 10))   # near concentration

    features.append(np.std(az)) # Angular spread
    features.append(np.mean(np.abs(np.diff(np.sort(az))))) #Average spacing between angles => how regularly points are distributed

    features.append(len(d) / 9000) # normalised density (assuming max 9000 points)

    features.append(np.std(d)) # Spread of distances
    features.append(np.var(d)) # Same as std but squared => more penalty for outliers

    

    # features.append(abs(front_left_mean - front_right_mean)) 
    # features.append(abs(front_mean - back_mean))
    return np.array(features)


def extract_frame_features(regime_dir, fid):
    radar_front = regime_dir / "ego" / "radar_front"
    radar_back = regime_dir / "ego" / "radar_back"
    cam_dir = regime_dir / "ego" / "camera_front"

    radar_feats = []
    paths = [
        radar_front / f"left_{fid}.npy",
        radar_front / f"right_{fid}.npy",
        radar_back / f"left_{fid}.npy",
        radar_back / f"right_{fid}.npy"
    ]

    means = []
    for p in paths:
        if not p.exists():
            return None
        feats = extract_radar_features(p)
        radar_feats.append(feats)
        means.append(feats[1])

    radar_feats = np.concatenate(radar_feats)

    if len(means) == 4:
        front_mean = (means[0] + means[1]) / 2
        back_mean = (means[2] + means[3]) / 2

        radar_feats = np.append(radar_feats, abs(front_mean - back_mean))
        radar_feats = np.append(radar_feats, abs(means[0] - means[1]))

    cam_path = cam_dir / f"{fid}.png"
    if not cam_path.exists():
        return None

    cam_feats = extract_camera_features(cam_path)

    return np.concatenate([radar_feats, cam_feats])  # 56 + 10 = 66


# ---------- DATASET ----------

def build_dataset(run_dirs):
    X, y = [], []

    for run_dir in run_dirs:
        for regime in ["clear", "fog", "rain"]:
            regime_dir = run_dir / regime
            radar_front = regime_dir / "ego" / "radar_front"

            if not radar_front.exists():
                continue
            
            prev_mean = None
            for f in radar_front.glob("left_*.npy"):
                fid = f.stem.split("_")[-1]

                feat = extract_frame_features(regime_dir, fid)

                if feat is None:
                    continue
                curr_mean = feat[1]
                temporal = 0 if prev_mean is None else abs(curr_mean - prev_mean)
                prev_mean = curr_mean
                
                feat = np.append(feat, temporal)

                X.append(feat)
                y.append(REGIME_MAP[regime])

    return np.array(X), np.array(y)


# ---------- MODEL ----------

class MLP(nn.Module):
    def __init__(self, in_dim=97):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)


# ---------- TRAIN ----------

def train():

    # get runs
    runs = sorted(DATA_ROOT.glob("run_*"))
    # random.shuffle(runs)

    split = int(0.8 * len(runs))
    train_runs = runs[:split]
    val_runs = runs[split:]
    # print("TRAIN:", [r.name for r in train_runs])
    # print("VAL:", [r.name for r in val_runs])
    # build datasets
    X_train, y_train = build_dataset(train_runs)
    X_val, y_val = build_dataset(val_runs)

    print("Train:", X_train.shape, "Val:", X_val.shape)

    # normalize using TRAIN stats only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6

    np.save(MODEL_DIR / "mean.npy", mean)
    np.save(MODEL_DIR / "std.npy", std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # to torch
    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    # Shuffle labels to test if model was learning signal(~33% expected) => if still high, then data leakage likely
    # perm = np.random.permutation(len(y_train))
    # y_train = y_train[perm]

    for epoch in range(25):
        perm = torch.randperm(X_train.size(0), device=X_train.device)
        X_train = X_train[perm]
        y_train = y_train[perm]

        model.train()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_val).argmax(dim=1)
            # Remove camera
            # X_val_copy = X_val.clone()
            # X_val_copy[:, 86:96] = 0
            # pred = model(X_val_copy).argmax(dim=1)

            # Remove radar
            # X_val_copy = X_val.clone()
            # X_val_copy[:, :86] = 0      # radar
            # X_val_copy[:, 96] = 0       # temporal
            # pred = model(X_val_copy).argmax(dim=1)

            acc = (pred == y_val).float().mean().item()
        
        with torch.no_grad():
            pred_train = model(X_train).argmax(dim=1)
            acc_train = (pred_train == y_train).float().mean().item()

        print(f"Epoch {epoch:02d} | Loss {loss.item():.4f} | Val Acc {acc:.3f} | Train Acc {acc_train:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    
    print(f"Best Val Acc: {best_acc:.3f}")


if __name__ == "__main__":
    train() 