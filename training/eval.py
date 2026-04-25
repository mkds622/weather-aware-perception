import torch
import numpy as np
from dataset import WeatherDataset
from model import MLP
from utils import split_runs

ROOT = "weather_dataset_extensive"

_, val_runs = split_runs(ROOT)
val_ds = WeatherDataset(ROOT, val_runs)

# load trained model
model = MLP(len(val_ds[0][0]), len(val_ds[0][1])).cuda()
model.load_state_dict(torch.load("model.pth"))

errors = []

model.eval()
with torch.no_grad():
    for x, y in val_ds:
        x = torch.tensor(x).float().cuda()
        y = torch.tensor(y).float().cuda()

        pred = model(x)
        errors.append((pred - y).cpu().numpy())

errors = np.array(errors)

# overall regression error
print("Mean abs error:", np.mean(np.abs(errors)))