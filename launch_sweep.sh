#!/bin/bash
# Launch W&B hyperparameter sweep

set -e

cd /home/nicolas/Github/ml4cv_project
source venv/bin/activate

echo "================================================================"
echo "W&B Hyperparameter Sweep - YOLOv11n Cone Detection"
echo "================================================================"
echo ""
echo "Baseline: mAP50 = 0.714"
echo "Target: mAP50 > 0.80 (12% improvement)"
echo ""
echo "Search method: Bayesian optimization"
echo "Epochs per run: 100 (for quick evaluation)"
echo "Early termination: Enabled (Hyperband)"
echo ""
echo "================================================================"
echo ""

# Initialize sweep and get sweep ID
echo "Initializing W&B sweep..."
SWEEP_ID=$(wandb sweep sweep_config.yaml 2>&1 | grep "wandb agent" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to create sweep"
    exit 1
fi

echo "Sweep created: $SWEEP_ID"
echo ""
echo "Starting sweep agent..."
echo "The agent will run continuously, launching training runs automatically."
echo "Press Ctrl+C to stop the sweep agent."
echo ""
echo "Monitor progress at: https://wandb.ai/ncridlig-ml4cv/runs-sweep"
echo ""
echo "================================================================"
echo ""

# Run sweep agent
# The agent will continuously pull new configs and train until:
# - You stop it with Ctrl+C
# - The sweep completes all runs
wandb agent "$SWEEP_ID"
