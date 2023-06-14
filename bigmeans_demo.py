# Big-means: K-means for big data clustering

# Rustam Mussabayev, Nenad Mladenovic, Bassem Jarboui, Ravil Mussabayev. How to Use K-means for Big Data Clustering? // Pattern Recognition, Volume 137, May 2023, 109269; https://doi.org/10.1016/j.patcog.2022.109269

from bigmeans import *
import math
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    filename = 'skin_segmentation.data'
    columns = slice(0, 3)
    with open(filename, newline='') as f1:
        reader = csv.reader(f1, delimiter='\t')
        raw = [row[columns] for row in reader]
    return np.array(raw, dtype=float)

points = load_dataset()


# Big-means parameters:
sample_size = 160000 # The number of data points to be randomly selected from the input dataset at each iteration of the Big-means.
n_centers = 25 # The desired number of clusters
max_iter = 2000 # Maximum number of samples to be processed
tmax = 1000.0 # The time limit for the search process (in seconds); a zero or negative value means no limit.
local_max_iters = 300 # The maximum number of K-means iterations before declaring convergence and stopping the clustering process for each sample.
local_tol = 0.0001 # The threshold below which the relative change in the objective function between two iterations must fall to declare convergence of K-means.
n_candidates = 3 # The number of candidate centers to choose from at each stage of the K-means++ initialization algorithm



nb.set_num_threads(nb.config.NUMBA_NUM_THREADS) # Set the number of threads for parallel execution to the maximum possible
#nb.set_num_threads(3) # Set the number of threads for parallel execution to the some value

# Best Known Solution (for comparison)
f_best = 102280000


print('SEQUENTIAL BIG-MEANS:')
print()
centers, objective, assignment, n_iter, best_n_iter, best_time, n_dists = big_means_sequential(points, n_centers, sample_size, max_iter, tmax, local_max_iters, local_tol, n_candidates, True)
print()
print('#Iterations: ', n_iter)
print('#Distances: ', n_dists)
print('Full Objective: ', objective)
objective_gap = round((objective - f_best) / objective * 100, 2)
print('Objective Gap: ', objective_gap, '%')
print()


print("BIG-MEANS WITH 'INNER PARALLELISM':")
print()
centers, objective, assignment, n_iter, best_n_iter, best_time, n_dists = big_means_inner(points, n_centers, sample_size, max_iter, tmax, local_max_iters, local_tol, n_candidates, True)
print()
print('#Iterations: ', n_iter)
print('#Distances: ', n_dists)
print('Full Objective: ', objective)
objective_gap = round((objective - f_best) / objective * 100, 2)
print('Objective Gap: ', objective_gap, '%')
print()


print("BIG-MEANS WITH 'COMPETITIVE PARALLELISM':")
print()
centers, objective, assignment, n_iter, best_n_iter, best_time, n_dists = big_means_competitive(points, n_centers, sample_size, max_iter, tmax, local_max_iters, local_tol, n_candidates, True)
print()
print('#Iterations: ', n_iter)
print('#Distances: ', n_dists)
print('Full Objective: ', objective)
objective_gap = round((objective - f_best) / objective * 100, 2)
print('Objective Gap: ', objective_gap, '%')
print()


print("BIG-MEANS WITH 'COLLECTIVE PARALLELISM':")
print()
centers, objective, assignment, n_iter, best_n_iter, best_time, n_dists = big_means_collective(points, n_centers, sample_size, max_iter, tmax, local_max_iters, local_tol, n_candidates, True)
print()
print('#Iterations: ', n_iter)
print('#Distances: ', n_dists)
print('Full Objective: ', objective)
objective_gap = round((objective - f_best) / objective * 100, 2)
print('Objective Gap: ', objective_gap, '%')
print()


print("BIG-MEANS WITH 'HYBRID PARALLELISM':")
print()
centers, objective, assignment, n_iter, best_n_iter, best_time, n_dists = big_means_hybrid(points, n_centers, sample_size, max_iter, max_iter, tmax, tmax, local_max_iters, local_tol, n_candidates, True)
print()
print('#Iterations: ', n_iter)
print('#Distances: ', n_dists)
print('Full Objective: ', objective)
objective_gap = round((objective - f_best) / objective * 100, 2)
print('Objective Gap: ', objective_gap, '%')
print()
