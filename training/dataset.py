import os
import json
import numpy as np
from features import lidar_features, radar_features


class WeatherDataset:
    def __init__(self, root, runs):
        self.samples = []

        # iterate over selected runs
        for run in runs:
            run_path = os.path.join(root, run)
            labels_path = os.path.join(run_path, "labels")

            for f in os.listdir(labels_path):
                fid = f.replace(".json", "")

                # store paths for each frame
                self.samples.append({
                    "lidar": os.path.join(run_path, "ego_1/lidar", f"{fid}.bin"),
                    "radar_f": os.path.join(run_path, "ego_1/radar_front", f"{fid}.npy"),
                    "radar_b": os.path.join(run_path, "ego_1/radar_back", f"{fid}.npy"),
                    "label": os.path.join(labels_path, f)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # extract features from each modality
        x1 = lidar_features(s["lidar"])
        x2 = radar_features(s["radar_f"])
        x3 = radar_features(s["radar_b"])

        x = np.concatenate([x1, x2, x3])  # final feature vector

        # load weather label
        with open(s["label"]) as f:
            y = json.load(f)["weather"]

        keys = [
            "cloudiness",
            "precipitation",
            "precipitation_deposits",
            "wind_intensity",
            "fog_density",
            "fog_distance",
            "fog_falloff",
            "sun_azimuth_angle",
            "sun_altitude_angle"
        ]

        y = np.array([y[k] for k in keys], dtype=np.float32)

        # normalize to ~0–1
        y = y / np.array([100, 100, 100, 50, 100, 100, 2, 360, 90], dtype=np.float32)

        return x, y