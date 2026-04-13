#!/usr/bin/env bash
set -e

source ~/projects/RAWAP_carla_l4dr/.venv/bin/activate
cd ~/projects/RAWAP_carla_l4dr/carla_l4dr
python scripts/collect_two_cars.py
