import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = Path("/home/meet/projects/RAWAP_carla_l4dr/weather-aware-perception/raw/weather_dataset_extended_2")
MODEL_DIR = Path("models/carla/radar_only_extended_2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REGIME_MAP = {"clear": 0, "fog": 1, "rain": 2}


# ---------- RADAR FEATURES (UPDATED) ----------

def extract_radar_features(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()

    d = data["depth"]
    az = data["azimuth"]

    if len(d) == 0:
        return np.zeros(2)

    features = []

    # density
    # features.append(len(d))

    # distance stats
    # features.extend([d.mean(), d.std(), d.min(), d.max()])

    # azimuth bins (4)
    # bins = [-np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3]
    # features.extend(np.histogram(az, bins=bins)[0])

    # depth histogram (5)
    # depth_bins = np.linspace(0, 100, 6)
    # features.extend(np.histogram(d, bins=depth_bins)[0])

    # weather-aware
    # features.append(np.mean(d > 50))
    # features.append(np.mean(d < 10))

    features.append(np.std(az))
    features.append(np.mean(np.abs(np.diff(np.sort(az)))))

    # features.append(len(d) / 9000)

    # features.append(np.std(d))
    # features.append(np.var(d))

    return np.array(features)


# ---------- FRAME FEATURES (UPDATED: multi + temporal) ----------

def extract_frame_features(regime_dir, fid, prev_mean):
    radar_front = regime_dir / "ego" / "radar_front"
    radar_back = regime_dir / "ego" / "radar_back"

    paths = [
        radar_front / f"left_{fid}.npy",
        radar_front / f"right_{fid}.npy",
        radar_back / f"left_{fid}.npy",
        radar_back / f"right_{fid}.npy"
    ]

    feats = []
    means = []

    for p in paths:
        if not p.exists():
            return None, prev_mean

        f = extract_radar_features(p)
        feats.append(f)
        means.append(f[1])  # mean depth

    radar_feats = np.concatenate(feats)

    # multi-radar
    front_mean = (means[0] + means[1]) / 2
    back_mean = (means[2] + means[3]) / 2

    # radar_feats = np.append(radar_feats, abs(front_mean - back_mean))
    # radar_feats = np.append(radar_feats, abs(means[0] - means[1]))

    # temporal
    temporal = 0 if prev_mean is None else abs(front_mean - prev_mean)
    prev_mean = front_mean

    # radar_feats = np.append(radar_feats, temporal)

    return radar_feats, prev_mean


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

                feat, prev_mean = extract_frame_features(regime_dir, fid, prev_mean)

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
            nn.Linear(8, 128),  # 4*21 + 2 + 1 = 87
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
    # random.shuffle(runs)  # keep deterministic

    split = int(0.8 * len(runs))
    train_runs = runs[:split]
    val_runs = runs[split:]

    # print("TRAIN:", [r.name for r in train_runs])
    # print("VAL:", [r.name for r in val_runs])

    X_train, y_train = build_dataset(train_runs)
    X_val, y_val = build_dataset(val_runs)

    print("Train:", X_train.shape, "Val:", X_val.shape)

    # normalize
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

    # ---------- TEST HOOKS ----------

    # Test: shuffle labels (leak check)
    # perm = np.random.permutation(len(y_train))
    # y_train = y_train[perm]

    for epoch in range(25):

        # shuffle batch
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

        print(f"Epoch {epoch:02d} | Loss {loss.item():.4f} | Val Acc {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_DIR / "model.pt")

    print(f"Best Val Acc: {best_acc:.3f}")


if __name__ == "__main__":
    train()