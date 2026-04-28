"""
Camera-only weather classification using CARLA data.

- Input: camera_front/*.png
- Features: grayscale statistics, contrast, edge variance
- Output: 3-class weather classification (clear, rain, fog)

"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random
import cv2
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = Path("/home/meet/projects/RAWAP_carla_l4dr/weather-aware-perception/raw/weather_dataset_extended_2")
MODEL_DIR = Path("models/carla/camera_only")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REGIME_MAP = {"clear": 0, "fog": 1, "rain": 2}


# ---------- FEATURES ----------

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
        cv2.Laplacian(gray, cv2.CV_64F).var(),
        np.mean(gray < 50),
        np.mean(gray > 200)
    ])


def extract_frame_features(regime_dir, fid):
    cam_path = regime_dir / "ego" / "camera_front" / f"{fid}.png"
    if not cam_path.exists():
        return None
    return extract_camera_features(cam_path)


# ---------- DATASET ----------

def build_dataset(run_dirs):
    X, y = [], []

    for run_dir in run_dirs:
        for regime in ["clear", "fog", "rain"]:
            regime_dir = run_dir / regime
            cam_dir = regime_dir / "ego" / "camera_front"

            if not cam_dir.exists():
                continue

            for f in cam_dir.glob("*.png"):
                fid = f.stem
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
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
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

    np.save(MODEL_DIR / "mean.npy", mean)
    np.save(MODEL_DIR / "std.npy", std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(25):
        perm = torch.randperm(X_train.size(0), device=DEVICE)
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