#!/bin/bash
# Behavior Cloning Data Collection Script
# Collects expert demonstrations by running expert_collector vs python-algo

set -e

# Configuration
NUM_EPISODES=${1:-100}
OUTPUT_DIR="${2:-bc_data}"
PARENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "================================"
echo "Expert Demonstration Collection"
echo "================================"
echo "Episodes: $NUM_EPISODES"
echo "Output Directory: $OUTPUT_DIR"
echo "Parent Directory: $PARENT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Paths
COLLECTOR_DIR="bc"
ALGO_DIR="python-algo-weak"

# Run episodes
for i in $(seq 1 $NUM_EPISODES); do
    echo "--------------------------------"
    echo "Episode $i / $NUM_EPISODES"
    echo "--------------------------------"
    
    # Set environment variables for this episode
    export GAME_NUM=$i
    export BC_OUTPUT_DIR="$OUTPUT_DIR"
    
    # Run match: collector vs python-algo
    cd "$PARENT_DIR"
    echo "$PARENT_DIR"
    python scripts/run_match.py "$COLLECTOR_DIR" "$ALGO_DIR"
    
    echo "Episode $i completed"
    echo ""
done

# echo "=================================="
# echo "Collection Complete!"
# echo "=================================="
# echo "Collected $NUM_EPISODES episodes in $OUTPUT_DIR"
# echo ""
# echo "Files:"
# ls -lh "$OUTPUT_DIR"/*.pkl | head -10
