# Big-means: K-means for big data clustering
# Original paper:
# Rustam Mussabayev, Nenad Mladenovic, Bassem Jarboui, Ravil Mussabayev. How to Use K-means for Big Data Clustering? // Pattern Recognition, Volume 137, May 2023, 109269; 
# https://doi.org/10.1016/j.patcog.2022.109269

# BIG-MEANS PARAMETERS:
# sample_size : The number of data points to be randomly selected from the input dataset at each iteration of the Big-means.
# n_centers : The desired number of clusters
# max_iter : Maximum number of samples to be processed
# tmax : The time limit for the search process (in seconds); a zero or negative value means no limit.
# local_max_iters : The maximum number of K-means iterations before declaring convergence and stopping the clustering process for each sample.
# local_tol : The threshold below which the relative change in the objective function between two iterations must fall to declare convergence of K-means.
# n_candidates : The number of candidate centers to choose from at each stage of the K-means++ initialization algorithm.

import math
import time
import numpy as np
import numba as nb
from numba import njit, prange, objmode


@njit(parallel = False)
def distance_mat(X,Y):
    out = np.dot(X, Y.T)
    NX = np.sum(X*X, axis=1)
    NY = np.sum(Y*Y, axis=1)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            out[i,j] = NX[i] - 2.*out[i,j] + NY[j]
    return out


@njit(parallel = True)
def distance_mat_parallel(X, Y):
    X_rows, X_cols = X.shape
    Y_rows, Y_cols = Y.shape
    out = np.zeros((X_rows, Y_rows))
    NX = np.sum(X*X, axis=1)
    NY = np.zeros(Y_rows)
    for i in prange(Y_rows):
        for j in range(Y_cols):
            NY[i] += Y[i, j] * Y[i, j]        
        for j in range(X_rows):        
            for k in range(X_cols):
                out[j, i] += X[j, k] * Y[i, k]
            out[j, i] = NX[j] - 2 * out[j, i] + NY[i]
    return out


@njit(parallel = False)
def kmeans_plus_plus(points, centers, n_new_centers=2, n_candidates=6):
    n_points, n_features = points.shape
    n_centers = centers.shape[0]
    n_dists = 0
    center_inds = np.full(n_new_centers, -1)
    if n_points > 0 and n_features > 0 and n_new_centers > 1 and n_candidates > 0:
        if n_centers == 0:
            center_inds[0] = np.random.randint(n_points)
            closest_dist_sq = distance_mat(points[center_inds[0:1]], points)[0]
            n_dists += n_points
            n_added_centers = 1
        else:           
            dist_mat = distance_mat(centers, points)
            n_dists += n_centers * n_points
            closest_dist_sq = np.empty(n_points)
            for j in range(n_points):
                min_dist = dist_mat[0, j]
                for i in range(1, n_centers):
                    if dist_mat[i, j] < min_dist:
                        min_dist = dist_mat[i, j]
                closest_dist_sq[j] = min_dist
            n_added_centers = 0
        current_pot = np.sum(closest_dist_sq)
        for c in range(n_added_centers, n_new_centers):
            rand_vals = np.random.random_sample(n_candidates) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
            dists = distance_mat(points[candidate_ids], points)
            n_dists += dists.size
            dists = np.minimum(dists, closest_dist_sq)
            candidates_pot = np.sum(dists, axis=1)
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = dists[best_candidate]
            center_inds[c] = candidate_ids[best_candidate]
    return center_inds, n_dists


@njit(parallel = True)
def kmeans_plus_plus_parallel(points, centers, n_new_centers=2, n_candidates=6):
    n_points, n_features = points.shape
    n_centers = centers.shape[0]
    n_dists = 0
    center_inds = np.full(n_new_centers, -1)
    if n_points > 0 and n_features > 0 and n_new_centers > 1 and n_candidates > 0:
        if n_centers == 0:
            center_inds[0] = np.random.randint(n_points)
            closest_dist_sq = distance_mat_parallel(points[center_inds[0:1]], points)[0]
            n_dists += n_points
            n_added_centers = 1
        else:           
            dist_mat = distance_mat_parallel(centers, points)
            n_dists += n_centers * n_points
            closest_dist_sq = np.empty(n_points)
            for j in prange(n_points):
                min_dist = dist_mat[0, j]
                for i in range(1, n_centers):
                    if dist_mat[i, j] < min_dist:
                        min_dist = dist_mat[i, j]
                closest_dist_sq[j] = min_dist
            n_added_centers = 0
        current_pot = np.sum(closest_dist_sq)
        for c in range(n_added_centers, n_new_centers):
            rand_vals = np.random.random_sample(n_candidates) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
            dists = distance_mat_parallel(points[candidate_ids], points)
            n_dists += dists.size
            dists = np.minimum(dists, closest_dist_sq)
            candidates_pot = np.sum(dists, axis=1)
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = dists[best_candidate]
            center_inds[c] = candidate_ids[best_candidate]
    return center_inds, n_dists


@njit(parallel = False)
def kmeans(points, centers, max_iters = -1, tol=0.0, use_inner_product = True):
    def dist2(point1, point2):
        if use_inner_product:
            s1 = s2 = s3 = 0.0
            for i in range(point1.shape[0]):
                s1 += point1[i]*point1[i]
                s2 += point1[i]*point2[i]
                s3 += point2[i]*point2[i]
            return s1 - 2*s2 + s3            
        else:
            d = 0.0
            for i in range(point1.shape[0]):
                d += (point1[i]-point2[i])**2
            return d   
    assert points.ndim == 2
    m, n = points.shape  
    assert (centers.ndim == 2) and (centers.shape[1] == n)
    k = centers.shape[0]
    assignment = np.full(m, -1)
    center_sums = np.empty((k, n))
    center_counts = np.zeros(k)
    f = np.inf
    n_iters = 0
    if (m > 0) and (n > 0) and (k > 0):
        objective_previous = np.inf
        tolerance = np.inf
        while True:            
            f = 0.0 # assignment step
            n_changed = 0
            for i in range(m):
                min_d = np.inf
                min_ind = -1
                for j in range(k):
                    d = dist2(points[i], centers[j])
                    if d < min_d:
                        min_d = d
                        min_ind = j
                if assignment[i] != min_ind:
                    n_changed += 1
                    assignment[i] = min_ind
                f += min_d
            n_iters += 1
            tolerance = 1 - f/objective_previous
            objective_previous = f
            
            if ((max_iters >= 0) and (n_iters >= max_iters)) or (n_changed == 0) or ((tol > 0.0) and (tolerance <= tol)):
                break
            for i in range(k): # update step
                center_counts[i] = 0.0
                for j in range(n):
                    center_sums[i,j] = 0.0
                    centers[i,j] = np.nan
            for i in range(m):
                center_ind = assignment[i]
                if center_ind > -1:
                    for j in range(n):
                        center_sums[center_ind,j] += points[i,j]
                    center_counts[center_ind] += 1.0                    
            for i in range(k):
                if center_counts[i] > 0.0:
                    for j in range(n):
                        centers[i,j] = center_sums[i,j] / center_counts[i]
    return f, n_iters, assignment, n_iters*k*m


@njit(parallel = True)
def kmeans_parallel(points, centers, max_iters = -1, tol=0.0, use_inner_product = True):
    def dist2(point1, point2):
        if use_inner_product:
            s1 = s2 = s3 = 0.0
            for i in range(point1.shape[0]):
                s1 += point1[i]*point1[i]
                s2 += point1[i]*point2[i]
                s3 += point2[i]*point2[i]
            return s1 - 2*s2 + s3            
        else:
            d = 0.0
            for i in range(point1.shape[0]):
                d += (point1[i]-point2[i])**2
            return d   
    assert points.ndim == 2
    m, n = points.shape  
    assert (centers.ndim == 2) and (centers.shape[1] == n)
    k = centers.shape[0]
    assignment = np.full(m, -1)
    center_sums = np.empty((k, n))
    center_counts = np.zeros(k)
    f = np.inf
    n_iters = 0
    if (m > 0) and (n > 0) and (k > 0):
        objective_previous = np.inf
        tolerance = np.inf
        while True:            
            f = 0.0 # assignment step
            n_changed = 0
            for i in prange(m):
                min_d = np.inf
                min_ind = -1
                for j in range(k):
                    d = dist2(points[i], centers[j])
                    if d < min_d:
                        min_d = d
                        min_ind = j
                if assignment[i] != min_ind:
                    n_changed += 1
                    assignment[i] = min_ind
                f += min_d
            n_iters += 1
            tolerance = 1 - f/objective_previous
            objective_previous = f
            
            if ((max_iters >= 0) and (n_iters >= max_iters)) or (n_changed == 0) or ((tol > 0.0) and (tolerance <= tol)):
                break
            for i in range(k): # update step
                center_counts[i] = 0.0
                for j in range(n):
                    center_sums[i,j] = 0.0
                    centers[i,j] = np.nan
            for i in range(m):
                center_ind = assignment[i]
                if center_ind > -1:
                    for j in range(n):
                        center_sums[center_ind,j] += points[i,j]
                    center_counts[center_ind] += 1.0                    
            for i in range(k):
                if center_counts[i] > 0.0:
                    for j in range(n):
                        centers[i,j] = center_sums[i,j] / center_counts[i]
    return f, n_iters, assignment, n_iters*k*m


# Sequential Big-means
@njit(parallel = False)
def big_means_sequential(points, n_centers = 3, sample_size = 100, max_iter = 10000, tmax = 10.0, local_max_iters=300, local_tol=0.0001, n_candidates = 3, printing=False):
    n_points, n_features = points.shape
    assert sample_size <= n_points
    if printing:
        with objmode:
            print ('%-30s%-15s%-15s' % ('sample objective', 'n_iter', 'cpu_time'))
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
    cpu_time = 0.0

    centers = np.full((n_centers, n_features), np.nan)
    objective = np.inf
    n_dists = 0
    n_iter = 0
    best_time = 0.0
    best_n_iter = 0
    while (n_iter < max_iter or max_iter <= 0) and (cpu_time < tmax or tmax <= 0.0):
        sample = points[np.random.choice(n_points, sample_size, replace=False)]
        new_centers = np.copy(centers)
        degenerate_mask = np.sum(np.isnan(new_centers), axis = 1) > 0
        n_degenerate = np.sum(degenerate_mask)
        if n_degenerate > 0:
            center_inds, num_dists = kmeans_plus_plus(sample, new_centers[~degenerate_mask], n_degenerate, n_candidates)
            n_dists += num_dists
            new_centers[degenerate_mask,:] = sample[center_inds,:]
        new_objective, _, _, num_dists = kmeans(sample, new_centers, local_max_iters, local_tol, True)
        n_dists += num_dists
        with objmode(cpu_time = 'float64'):
            cpu_time = time.perf_counter() - start_time
        n_iter += 1
        if new_objective < objective:
            objective = new_objective
            centers = np.copy(new_centers)
            if printing:
                with objmode:
                    print ('%-30f%-15i%-15.2f' % (objective, n_iter, cpu_time))
            best_time = cpu_time
            best_n_iter = n_iter    
        
    # When 'max_iters = 0' is used for kmeans, only the assignment step will be performed
    full_objective, _, assignment, num_dists = kmeans(points, centers, 0, local_tol, True)
    n_dists += num_dists
    return centers, full_objective, assignment, n_iter, best_n_iter, best_time, n_dists
                     

    
# Big-means with "Inner Parallelism":
# Separate data samples are clustered sequentially one-by-one, but the clustering process itself 
# is parallelized on the level of internal implementation of the K-means and K-means++ functions.
@njit(parallel = True)
def big_means_inner(points, n_centers = 3, sample_size = 100, max_iter = 10000, tmax = 10.0, local_max_iters=300, local_tol=0.0001, n_candidates = 3, printing=False):
    n_points, n_features = points.shape
    assert sample_size <= n_points
    if printing:
        with objmode:
            print ('%-30s%-15s%-15s' % ('sample objective', 'n_iter', 'cpu_time'))
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
    cpu_time = 0.0

    centers = np.full((n_centers, n_features), np.nan)
    objective = np.inf
    n_dists = 0
    n_iter = 0
    best_time = 0.0
    best_n_iter = 0
    while (n_iter < max_iter or max_iter <= 0) and (cpu_time < tmax or tmax <= 0.0):
        sample = points[np.random.choice(n_points, sample_size, replace=False)]
        new_centers = np.copy(centers)
        degenerate_mask = np.sum(np.isnan(new_centers), axis = 1) > 0
        n_degenerate = np.sum(degenerate_mask)
        if n_degenerate > 0:
            center_inds, num_dists = kmeans_plus_plus_parallel(sample, new_centers[~degenerate_mask], n_degenerate, n_candidates)
            n_dists += num_dists
            new_centers[degenerate_mask,:] = sample[center_inds,:]
        

        new_objective, _, _, num_dists = kmeans_parallel(sample, new_centers, local_max_iters, local_tol, True)
        n_dists += num_dists
        with objmode(cpu_time = 'float64'):
            cpu_time = time.perf_counter() - start_time
        n_iter += 1
        if new_objective < objective:
            objective = new_objective
            centers = np.copy(new_centers)
            if printing:
                with objmode:
                    print ('%-30f%-15i%-15.2f' % (objective, n_iter, cpu_time))
            best_time = cpu_time
            best_n_iter = n_iter           
       
    # When 'max_iters = 0' is used for kmeans, only the assignment step will be performed
    full_objective, _, assignment, num_dists = kmeans_parallel(points, centers, 0, 0.0, True)
    n_dists += num_dists
    return centers, full_objective, assignment, n_iter, best_n_iter, best_time, n_dists


# Big-means with "Competitive Parallelism":
# Separate data samples are processed in parallel while each sample is clustered 
# on a separate CPU core using the regular / sequential implementations of 
# the K-means and K-means++ algorithms. Workers use only their previous own best centroids 
# for initialization at every iteration. This parallelization mode is called competitive since 
# all workers are independent and compete with each other.
@njit(parallel = True)
def big_means_competitive(points, n_centers = 3, sample_size = 100, max_iter = 10000, tmax = 10.0, local_max_iters=300, local_tol=0.0001, n_candidates = 3, printing=False):
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
        
    n_points, n_features = points.shape
    n_threads = nb.get_num_threads()
    assert sample_size <= n_points
    if printing:
        with objmode:
            print ('%-30s%-15s%-15s' % ('sample objective', 'n_iter', 'cpu_time'))
    cpu_time = 0.0

    centers = np.full((n_threads, n_centers, n_features), np.nan)
    objectives = np.full(n_threads, np.inf)
    n_dists = np.full(n_threads, 0)
    n_iters = np.full(n_threads, 0)
    running_time = np.full(n_threads, 0.0)
    best_times = np.full(n_threads, 0.0)
    best_n_iters = np.full(n_threads, 0)
    
    for t in prange(n_threads):
        while (np.sum(n_iters) < max_iter or max_iter <= 0) and (running_time[t] < tmax or tmax <= 0.0):        
            sample = points[np.random.choice(n_points, sample_size, replace=False)]
            best = np.argmin(objectives)
            best_objective = objectives[best]
            new_centers = centers[t].copy()
            degenerate_mask = np.sum(np.isnan(new_centers), axis = 1) > 0
            n_degenerate = np.sum(degenerate_mask)
            if n_degenerate > 0:
                center_inds, num_dists = kmeans_plus_plus(sample, new_centers[~degenerate_mask], n_degenerate, n_candidates)
                n_dists[t] += num_dists
                new_centers[degenerate_mask,:] = sample[center_inds,:]
                
            new_objective, _, _, num_dists = kmeans(sample, new_centers, local_max_iters, local_tol, True)
            n_dists[t] += num_dists
            with objmode(time_now = 'float64'):
                time_now = time.perf_counter() - start_time
            running_time[t] = time_now
            n_iters[t] += 1
            #if new_objective < objectives[t]:
            if new_objective < best_objective:
                objectives[t] = new_objective
                centers[t] = new_centers.copy()
                best_times[t] = time_now
                best_n_iters[t] = np.sum(n_iters)
                if printing:
                    with objmode:
                        print ('%-30f%-15i%-15.2f' % (new_objective, best_n_iters[t], time_now))
    
    best_ind = np.argmin(objectives)
    final_centers = centers[best_ind].copy()
        
    # When 'max_iters = 0' is used for kmeans, only the assignment step will be performed
    full_objective, _, assignment, full_num_dists = kmeans_parallel(points, final_centers, 0, 0.0, True)

    return final_centers, full_objective, assignment, np.sum(n_iters), best_n_iters[best_ind], best_times[best_ind], np.sum(n_dists)+full_num_dists


# Big-means with "Collective Parallelism":
# Separate data samples are processed in parallel while each sample is clustered 
# on a separate CPU core using the regular / sequential implementations 
# of the K-means and K-means++ algorithms. At all subsequent iterations, 
# each worker uses the best set of centroids among all workers obtained
# at previous iterations to initialize a new random data sample.
# This parallelization mode is called collective since 
# the workers share information about the best solutions.
@njit(parallel = True)
def big_means_collective(points, n_centers = 3, sample_size = 100, max_iter = 10000, tmax = 10.0, local_max_iters=300, local_tol=0.0001, n_candidates = 3, printing=False):
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
        
    n_points, n_features = points.shape
    n_threads = nb.get_num_threads()
    assert sample_size <= n_points
    if printing:
        with objmode:
            print ('%-30s%-15s%-15s' % ('sample objective', 'n_iter', 'cpu_time'))
    cpu_time = 0.0

    centers = np.full((n_threads, n_centers, n_features), np.nan)
    objectives = np.full(n_threads, np.inf)
    n_dists = np.full(n_threads, 0)
    n_iters = np.full(n_threads, 0)
    running_time = np.full(n_threads, 0.0)
    best_times = np.full(n_threads, 0.0)
    best_n_iters = np.full(n_threads, 0)
    
    for t in prange(n_threads):
        while (np.sum(n_iters) < max_iter or max_iter <= 0) and (running_time[t] < tmax or tmax <= 0.0):        
            sample = points[np.random.choice(n_points, sample_size, replace=False)]
            best = np.argmin(objectives)
            best_objective = objectives[best]
            new_centers = centers[best].copy()                            
            degenerate_mask = np.sum(np.isnan(new_centers), axis = 1) > 0
            n_degenerate = np.sum(degenerate_mask)
            if n_degenerate > 0:
                center_inds, num_dists = kmeans_plus_plus(sample, new_centers[~degenerate_mask], n_degenerate, n_candidates)
                n_dists[t] += num_dists
                new_centers[degenerate_mask,:] = sample[center_inds,:]
                
            new_objective, _, _, num_dists = kmeans(sample, new_centers, local_max_iters, local_tol, True)
            n_dists[t] += num_dists
            with objmode(time_now = 'float64'):
                time_now = time.perf_counter() - start_time
            running_time[t] = time_now
            n_iters[t] += 1
            if new_objective < best_objective:
                objectives[t] = new_objective
                centers[t] = new_centers.copy()
                best_times[t] = time_now
                best_n_iters[t] = np.sum(n_iters)
                if printing:
                    with objmode:
                        print ('%-30f%-15i%-15.2f' % (new_objective, best_n_iters[t], time_now))
    
    best_ind = np.argmin(objectives)
    final_centers = centers[best_ind].copy()
        
    # When 'max_iters = 0' is used for kmeans, only the assignment step will be performed
    full_objective, _, assignment, full_num_dists = kmeans_parallel(points, final_centers, 0, 0.0, True)

    return final_centers, full_objective, assignment, np.sum(n_iters), best_n_iters[best_ind], best_times[best_ind], np.sum(n_dists)+full_num_dists
