# Big-Means: A Simple and Effective Algorithm for Big Data Minimum Sum-of-Squares Clustering
# Original paper:
# Rustam Mussabayev, Nenad Mladenovic, Ravil Mussabayev, Bassem Jarboui. Big-means: Less is More for K-means Clustering. arXiv preprint arXiv:2204.07485. 14 Apr 2022. pp. 1-40
# https://arxiv.org/pdf/2204.07485.pdf
# rmusab@gmail.com


from bigmeans import *
import math
import numpy as np
import matplotlib.pyplot as plt


# Generation of synthetic dataset for clustering
grid_size = 6
n_features = 2
n_entities = 600000
standard_deviation = 0.06
n_candidates = 3
#nb.config.NUMBA_NUM_THREADS = 12 # The number of CPU cores to be used for parallel processing
entities, true_membership, true_centroids = generate_blobs_on_grid(n_entities, grid_size, n_features, standard_deviation)
if n_features == 2:
    draw_dataset(entities, true_centroids, true_membership)    
n_entities = entities.shape[0]
n_clusters = true_centroids.shape[0]


# Parameters of Big-means algorithm
parallel = True
sample_size = 3000
max_n_samples = 100000 # maximum number of samples for clustering
init_mode = 1 # initialization mode (0 - Forgy; 1 - K-means++)
tmax = 10 # time limit (in seconds)
local_max_iters = 300 # maximum number of iterations for K-means local search
local_tol = 0.0001 # relative tolerance for K-means local search
n_candidates = 3 # number of candidates for K-means++ initialization
printing = True # printing the intermediate result output


# Big-means clustering
centroids, objective, membership, n_iter, best_n_iter, best_time, n_dists = big_means_par1(entities, n_clusters, sample_size, max_n_samples, tmax, init_mode, local_max_iters, local_tol, n_candidates, parallel, printing)

print()
print('Final objective:', objective)
print()


# Visualization of clustering results
draw_dataset(entities, centroids, membership)






