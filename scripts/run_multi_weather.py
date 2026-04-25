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
from samplers import random_sampler, lhs_sampler

CONFIG_PATH = f"{configs.project_config['project_root']}/configs/sim_config.json"
BASE_OUTPUT = "weather_dataset"

USE_LHS = False

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
    runs_per_regime = 2

    run_id = 0

    for regime in regimes:
        if USE_LHS:
            weather_samples = lhs_sampler(regime, runs_per_regime)
        else: 
            weather_samples = [random_sampler(regime) for _ in range(runs_per_regime)]
        print(f'Weather samples created for ${regime}: ${str(weather_samples)}')
        for sample in weather_samples:
            update_config(sample, run_id)

            print(f"Running {run_id} | {regime}")
            run_collection()

            run_id += 1

if __name__ == "__main__":
    main()