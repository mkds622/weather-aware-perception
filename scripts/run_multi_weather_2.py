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
BASE_OUTPUT = "weather_dataset_extended_2"

USE_LHS = False

def update_config(weather_params, run_id, regime, spawn_index):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
        # print("loaded config", cfg);

    cfg["vehicle_1_spawn_index"] = spawn_index
    cfg["weather"]["mode"] = "custom"
    cfg["weather"]["params"] = weather_params
    cfg["output_root"] = f"{configs.project_config['project_root']}/raw/{BASE_OUTPUT}/run_{run_id:03d}/{regime}"

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

def run_collection():
    subprocess.run(["bash", f"{configs.project_config['project_root']}/scripts/run_collect_2.sh"], check=True)

def main():
    regimes = ["clear", "fog", "rain" ] 
    runs_per_regime = 25

    spawn_indices = [random.randint(0, 100) for _ in range(runs_per_regime)]

    run_id = 0

    for i in range(runs_per_regime):
        spawn_index = spawn_indices[i]
        for regime in regimes:
            if USE_LHS:
                weather_sample = lhs_sampler(regime, 1)[0]
            else: 
                weather_sample = random_sampler(regime)
            print(f'Weather sample for {regime}: {weather_sample}')

            update_config(weather_sample, run_id, regime, spawn_index)

            print(f"Running {run_id} | {regime}")
            run_collection()

        run_id += 1

if __name__ == "__main__":
    main()