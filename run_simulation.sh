#!/bin/bash

# Load config
CONFIG_FILE=${1:-config.ini}
DATA_DIR=$(grep "data_dir" $CONFIG_FILE | cut -d'=' -f2 | cut -d':' -f1 | xargs)

# Create data directory
mkdir -p $DATA_DIR

echo "Compiling with Pure Eigen (no external BLAS) + GSL + OpenMP..."

# Compile C++ code with Pure Eigen (no external BLAS dependencies)
g++ -std=c++14 -O3 -fopenmp \
    -DEIGEN_DONT_USE_MKL \
    -DEIGEN_DONT_USE_BLAS \
    -DEIGEN_DONT_USE_LAPACKE \
    -I/home/rik/Cpp_Libraries/eigen-master \
    -I src/include \
    src/main.cpp src/Simulation.cpp src/MatrixUtils.cpp \
    -lgsl -lgslcblas -lm \
    -o simulate

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful with Pure Eigen!"

    # Set number of threads
    export OMP_NUM_THREADS=8

    # Run simulation
    ./simulate $CONFIG_FILE

    echo "Simulation completed. Data stored in $DATA_DIR"
else
    echo "❌ Compilation failed!"
    exit 1
fi
