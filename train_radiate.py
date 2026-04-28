import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = Path("raw/radiate")
MODEL_DIR = Path("models/radiate/radiate_extensive")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REGIME_MAP = {"clear": 0, "rain": 1, "fog": 2}


# ---------- FEATURES ----------

def extract_radar_features(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = img.astype(np.float32)

    features = []

    # global stats
    features.extend([img.mean(), img.std(), img.min(), img.max()])

    # percentiles
    features.extend([
        np.percentile(img, 25),
        np.percentile(img, 50),
        np.percentile(img, 75)
    ])

    # density / intensity
    features.append(np.mean(img > 200))
    features.append(np.mean(img < 30))

    # total energy
    features.append(np.sum(img))

    # range bins (rows)
    row_bins = np.array_split(img, 5, axis=0)
    for b in row_bins:
        features.append(b.mean())

    # angular variation (columns)
    col_std = np.std(img, axis=0)
    features.append(col_std.mean())
    features.append(col_std.std())

    return np.array(features)  # ~18 dims


# ---------- DATA ----------

def load_split(split):
    X, y = [], []

    for regime in ["clear", "rain", "fog"]:
        polar_dir = DATA_ROOT / split / regime / "Navtech_Polar"

        if not polar_dir.exists():
            print("Missing:", polar_dir)
            continue

        for img_path in polar_dir.glob("*.png"):
            feat = extract_radar_features(img_path)
            if feat is None:
                continue

            X.append(feat)
            y.append(REGIME_MAP[regime])

    return np.array(X), np.array(y)


# ---------- MODEL ----------

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


# ---------- TRAIN ----------

def train():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print("Train:", X_train.shape, "Val:", X_val.shape)

    if len(X_train) == 0 or len(X_val) == 0:
        print("ERROR: dataset empty")
        return

    in_dim = X_train.shape[1]

    # normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    model = MLP(in_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0

    for epoch in range(20):
        # ----- TRAIN -----
        model.train()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_acc = (logits.argmax(1) == y_train).float().mean().item()

        # ----- VAL -----
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()

        print(f"Epoch {epoch:02d} | Loss {loss.item():.4f} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "model.pt")

    print("Best Val Acc:", best_val)


if __name__ == "__main__":
    train()