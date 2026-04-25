import sys
from pathlib import Path

# Add the project root to the python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from model import MLP
from utils import split_runs
import configs

ROOT = f"{configs.project_config['project_root']}/raw/weather_dataset_extensive"

# split by runs (prevents leakage)
train_runs, val_runs = split_runs(ROOT)

train_ds = WeatherDataset(ROOT, train_runs)
val_ds = WeatherDataset(ROOT, val_runs)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# infer input/output sizes
x0, y0 = train_ds[0]

model = MLP(len(x0), len(y0)).cuda()
opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    model.train()

    for x, y in train_loader:
        x = x.float().cuda()
        y = y.float().cuda()

        # normalize features per batch
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.float().cuda()
            y = y.float().cuda()

            x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

            pred = model(x)
            val_loss += loss_fn(pred, y).item()

    print(f"epoch {epoch} val_loss {val_loss:.4f}")