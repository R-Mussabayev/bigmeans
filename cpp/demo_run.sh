#!/bin/bash
set -e

#export OMP_NUM_THREADS=$(nproc)  # Use all available cores
export OMP_NUM_THREADS=80  # Use specific number of cores

./build/demo
