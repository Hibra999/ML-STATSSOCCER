#!/bin/bash
# Navigate to the directory of the script
cd "$(dirname "$0")"

# (Optional) Activate your virtual environment first:
# source "$HOME/python/envs/myenv/bin/activate"

# Run the local web application and pass any provided arguments.
python3 app.py "$@"
