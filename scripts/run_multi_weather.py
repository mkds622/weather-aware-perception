import sys
from pathlib import Path

# Add the project root to the python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import json
import os
import subprocess
import random

import configs

CONFIG_PATH = f"{configs.project_config['project_root']}/configs/sim_config.json"
BASE_OUTPUT = "weather_dataset_extensive_2"

def sample_weather(regime):
    if regime == "fog":
        return {
            "cloudiness": random.uniform(50, 100),
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "wind_intensity": random.uniform(0, 20),
            "sun_altitude_angle": random.uniform(10, 40),
            "fog_density": random.uniform(30, 90),
            "fog_distance": random.uniform(5, 50),
            "fog_falloff": random.uniform(0.5, 2.0),
            "wetness": 0.0
        }

    if regime == "rain":
        return {
            "cloudiness": random.uniform(60, 100),
            "precipitation": random.uniform(30, 100),
            "precipitation_deposits": random.uniform(30, 100),
            "wind_intensity": random.uniform(10, 50),
            "sun_altitude_angle": random.uniform(5, 30),
            "fog_density": random.uniform(0, 20),
            "fog_distance": random.uniform(50, 100),
            "fog_falloff": 1.0,
            "wetness": random.uniform(30, 100)
        }

    return {  # clear
        "cloudiness": random.uniform(0, 30),
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": random.uniform(0, 10),
        "sun_altitude_angle": random.uniform(30, 80),
        "fog_density": 0.0,
        "fog_distance": 100.0,
        "fog_falloff": 1.0,
        "wetness": 0.0
    }

def update_config(weather_params, run_id):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
        # print("loaded config", cfg);

    cfg["weather"]["mode"] = "custom"
    cfg["weather"]["params"] = weather_params
    cfg["output_root"] = f"{configs.project_config['project_root']}/raw/{BASE_OUTPUT}/run_{run_id:03d}"

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

def run_collection():
    subprocess.run(["bash", f"{configs.project_config['project_root']}/scripts/run_collect.sh"], check=True)

def main():
    regimes = ["clear", "fog", "rain" ] 
    runs_per_regime = 25

    run_id = 0

    for regime in regimes:
        for _ in range(runs_per_regime):
            weather = sample_weather(regime)
            update_config(weather, run_id)

            print(f"Running {run_id} | {regime}")
            run_collection()

            run_id += 1

if __name__ == "__main__":
    main()