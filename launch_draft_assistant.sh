#!/bin/bash

# Get the directory where the script is located (the project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- VIRTUAL ENVIRONMENT ---
# Activate the virtual environment located in the project root
source "$SCRIPT_DIR/.venv/bin/activate"

# --- PYTHONPATH ---
# Add the project's root directory to the PYTHONPATH
# This allows Python to find the 'src' module
export PYTHONPATH="$SCRIPT_DIR"

# --- CONFIGURATION ---
# TODO: Set your league's actual values here
LEAGUE_ID=328547063
TEAM_ID=8
FIRST_PICK=2

# --- EXECUTION ---
# Run the draft assistant script from the root directory
python3 "$SCRIPT_DIR/src/draft/run_draft_assistant.py" --league_id $LEAGUE_ID --team_id $TEAM_ID --first_pick $FIRST_PICK

echo "Draft assistant has been launched."