#!/usr/bin/env bash
set -e

source <local_path_to_src>/.venv/bin/activate
cd <local_path_to_src>/weather-aware-perception
python scripts/collect_two_cars_2.py
