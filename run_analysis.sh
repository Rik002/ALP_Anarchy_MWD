#!/bin/bash

# Load config
CONFIG_FILE=${1:-config.ini}
PLOT_DIR=$(grep "plot_dir" $CONFIG_FILE | cut -d'=' -f2 | cut -d':' -f1 | xargs)
PY_LIB_DIR=$(grep "py_lib_dir" $CONFIG_FILE | cut -d'=' -f2 | cut -d':' -f1 | xargs)

# Create directories
mkdir -p $PLOT_DIR
mkdir -p $(dirname $CONFIG_FILE)/data

echo "Checking if C++ simulation completed..."
python3 scripts/check_simulation.py $CONFIG_FILE
if [ $? -ne 0 ]; then
    echo "Please run the C++ simulation first: ./simulation $CONFIG_FILE"
    exit 1
fi

# Run data debugging
echo "Scanning data to find discrepancies..."
python3 scripts/debug_data.py $CONFIG_FILE
if [ $? -ne 0 ]; then
    echo "ERROR: Invalid data detected! Please fix the C++ simulation."
    exit 1
fi

# Run analysis
echo "Running Analysis Scripts..."
python3 scripts/analyze_data.py $CONFIG_FILE

# Run plotting
echo "generating Plot..."
python3 scripts/plot_results.py $CONFIG_FILE

echo "Analysis and plotting completed. Plots stored in $PLOT_DIR"
