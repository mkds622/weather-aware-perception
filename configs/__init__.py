import json
import os

# Get the absolute path to the JSON file relative to this script
_base_dir = os.path.dirname(__file__)
project_config_path = os.path.join(_base_dir, 'project_config.json')


with open(project_config_path, 'r', encoding='utf-8') as f:
    # This 'project_config' variable will be accessible via 'config.project_config'
    project_config = json.load(f)

import configs

print(configs.project_config)
