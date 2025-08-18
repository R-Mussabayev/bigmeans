# Big-means Clustering: A Simple and Effective Algorithm for Big Data Minimum Sum-of-Squares Clustering

An efficient parallel C++ implementation of the Big-means algorithm for Big Data clustering, leveraging Eigen and OpenMP.

The Big-Means algorithm was originally proposed and described in the following article:

Rustam Mussabayev, Nenad Mladenovic, Bassem Jarboui, Ravil Mussabayev. How to Use K-means for Big Data Clustering? // Pattern Recognition, Volume 137, May 2023, 109269; https://doi.org/10.1016/j.patcog.2022.109269

## Requirements
- C++17 or newer
- [Eigen3](https://eigen.tuxfamily.org/) (header-only)
- OpenMP (optional, for parallelism)

## Build
```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

## How to Run a Demo
1) Prepare your dataset (e.g., skin_segmentation.data).

2) (Optional) Set the number of OpenMP threads:
   export OMP_NUM_THREADS=$(nproc)         # Use all available cores
   export OMP_NUM_THREADS=16               # OpenMP threads number (example fixed value)

3) Run the program:
   ./build/demo