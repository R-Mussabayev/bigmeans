# Big-Means: A Simple and Effective Algorithm for Big Data Minimum Sum-of-Squares Clustering
# Original paper:
# Rustam Mussabayev, Nenad Mladenovic, Bassem Jarboui, Ravil Mussabayev. How to Use K-means for Big Data Clustering? // Pattern Recognition, Volume 137, May 2023, 109269; 
# https://doi.org/10.1016/j.patcog.2022.109269
# Rustam Mussabayev, Nenad Mladenovic, Ravil Mussabayev, Bassem Jarboui. Big-means: Less is More for K-means Clustering. arXiv preprint arXiv:2204.07485. 14 Apr 2022. pp. 1-40
# https://arxiv.org/pdf/2204.07485.pdf
# rmusab@gmail.com


import math
import time
import threading
import numpy as np
import numba as nb
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from numba import njit, prange, objmode, cuda


def normalization(X):
    X_min = np.amin(X, axis=0)
    X = X - X_min
    X_max = np.amax(X, axis=0)
    if X_max.ndim == 1:
        X_max[X_max == 0.0] = 1.0
    elif X_max.ndim == 0:
        if X_max == 0.0:
            X_max = 1.0
    else:
        X_max = 1.0
    X = X / X_max
    return X

# Generate isotropic Gaussian blobs
def gaussian_blobs(n_features = 2, n_samples = 1000, n_clusters = 5, cluster_std = 0.1):
    true_centers = np.random.rand(n_clusters, n_features)   
    X, labels = make_blobs(n_samples=n_samples, centers=true_centers, cluster_std=cluster_std)
    N = np.concatenate((true_centers,X))
    N = normalization(N)
    true_centers = N[:n_clusters]
    X = N[n_clusters:]
    return X, true_centers, labels


def draw_dataset(X, true_centers, original_labels, title = 'DataSet'):
    n_clusters = len(true_centers)
    plt.rcParams['figure.figsize'] = [10,10]
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    
    if original_labels.shape[0] == X.shape[0]:
        for k, col in zip(range(n_clusters), colors):
            my_members = original_labels == k

            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

            if true_centers.shape[0] > 0:
                cluster_center = true_centers[k]
                plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    else:
        plt.plot(X[:, 0], X[:, 1], '.')
    
    plt.title('DataSet')
    plt.show()

    
def generate_grid_nodes(coordinates1D, n_dimensions=3):
    coordinatesND = n_dimensions*[coordinates1D]
    mesh = np.array(np.meshgrid(*coordinatesND))
    grid_nodes = mesh.T.reshape(-1, n_dimensions)
    return grid_nodes


def generate_blobs_on_grid(n_samples=3000, grid_size=3, n_features=3, standard_deviation = 0.1):
    assert grid_size > 0
    cell_size = 1/grid_size
    half_cell_size = cell_size/2
    coordinates1D = np.linspace(half_cell_size, 1.0-half_cell_size, grid_size)
    true_centroids = generate_grid_nodes(coordinates1D, n_features)    
    samples, sample_membership = make_blobs(n_samples=n_samples, centers=true_centroids, cluster_std=standard_deviation)
    mask = np.all((samples >= 0.0) & (samples <= 1.0) , axis = 1)
    samples = samples[mask]
    sample_membership = sample_membership[mask]    
    return samples, sample_membership, true_centroids


def generate_blobs_on_grid(n_samples=3000, grid_size=3, n_features=3, standard_deviation = 0.1):
    assert grid_size > 0
    cell_size = 1/grid_size
    half_cell_size = cell_size/2
    coordinates1D = np.linspace(half_cell_size, 1.0-half_cell_size, grid_size)
    true_centroids = generate_grid_nodes(coordinates1D, n_features)
    n_centroids = true_centroids.shape[0]
    samples, sample_membership = make_blobs(n_samples=n_samples, centers=true_centroids, cluster_std=standard_deviation)    
    samples2 = np.concatenate((true_centroids, samples), axis=0)
    samples2 = normalization(samples2)
    true_centroids = samples2[:n_centroids]
    samples = samples2[n_centroids:]
    return samples, sample_membership, true_centroids


# choosing the new n additional centers for existing ones using the k-means++ logic
def _kmeanspp(samples, centroids, n_additional_centers=3, n_candidates=3):
    def cum_search(X, vals, out):
        n = X.shape[0]
        n_vals = vals.shape[0]
        assert n>0 and n_vals == out.shape[0] and n_vals>0
        cum_sum = 0.0
        ind_vals = 0
        sorted_inds = np.argsort(vals)
        for i in range(n):
            if not math.isnan(X[i]):
                cum_sum += X[i]
                while vals[sorted_inds[ind_vals]] <= cum_sum:
                    out[sorted_inds[ind_vals]] = i
                    ind_vals += 1
                    if ind_vals == n_vals:
                        return
        out[sorted_inds[ind_vals: n_vals]] = n-1       
    def distance_mat(X,Y):
        out = np.dot(X, Y.T)
        NX = np.empty(X.shape[0])
        for i in prange(X.shape[0]):
            s = 0.0
            for j in range(X.shape[1]):
                s += X[i,j]**2
            NX[i] = s
        NY = np.empty(Y.shape[0])
        for i in prange(Y.shape[0]):
            s = 0.0
            for j in range(Y.shape[1]):
                s += Y[i,j]**2
            NY[i] = s                
        for i in prange(X.shape[0]):
            for j in range(Y.shape[0]):
                out[i,j] = NX[i] - 2.*out[i,j] + NY[j]
        return out      
    n_samples, n_features = samples.shape
    n_centers = centroids.shape[0]
    n_dists = 0
    center_inds = np.full(n_additional_centers, -1)    
    nondegenerate_mask = np.sum(np.isnan(centroids), axis = 1) == 0
    n_nondegenerate_clusters = np.sum(nondegenerate_mask)    
    center_inds = np.full(n_additional_centers, -1)
    if (n_samples > 0) and (n_features > 0) and (n_additional_centers > 0):        
        if (n_candidates <= 0) or (n_candidates is None):
            n_candidates = 2 + int(np.log(n_nondegenerate_clusters+n_additional_centers))                                       
        if n_nondegenerate_clusters > 0:
            closest_dist_sq = np.full(n_samples, np.inf)            
            distances = distance_mat(centroids[nondegenerate_mask], samples)
            current_pot = 0.0
            for i in prange(n_samples):
                for j in range(n_nondegenerate_clusters):
                    closest_dist_sq[i] = min(distances[j,i],closest_dist_sq[i])
                current_pot += closest_dist_sq[i]
            n_added_centers = 0        
        else:
            center_inds[0] = np.random.randint(n_samples)
            indices = np.full(1, center_inds[0])
            distances = distance_mat(samples[indices], samples)
            closest_dist_sq = distances[0]
            current_pot = 0.0
            for i in prange(n_samples):
                current_pot += closest_dist_sq[i]
            n_added_centers = 1
        n_dists += distances.size
        candidate_ids = np.full(n_candidates, -1)
        candidates_pot = np.empty(n_candidates)
        for c in range(n_added_centers, n_additional_centers):
            rand_vals = np.random.random_sample(n_candidates) * current_pot
            cum_search(closest_dist_sq, rand_vals, candidate_ids)
            distances = distance_mat(samples[candidate_ids], samples)
            n_dists += distances.size
            for i in prange(n_candidates):
                candidates_pot[i] = 0.0
                for j in range(n_samples):
                    distances[i,j] = min(distances[i,j],closest_dist_sq[j])
                    candidates_pot[i] += distances[i,j]            
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]            
            for i in prange(n_samples):
                closest_dist_sq[i] = distances[best_candidate][i]
            center_inds[c] = candidate_ids[best_candidate]
    return center_inds, n_dists

kmeanspp = njit(parallel=False)(_kmeanspp)
kmeanspp_parallel = njit(parallel=True)(_kmeanspp)


@njit
def check_shapes(samples, sample_membership, centroids, centroid_sums, centroid_counts):
    assert samples.ndim == 2
    n_samples, n_features = samples.shape
    assert sample_membership.shape == (n_samples,)
    assert (centroids.ndim == 2) and (centroids.shape[1] == n_features)
    n_clusters = centroids.shape[0]
    assert centroid_sums.shape == (n_clusters, n_features)
    assert centroid_counts.shape == (n_clusters, )
    

@njit
def empty_state(n_samples, n_features, n_clusters):
    sample_membership = np.full(n_samples, -1)
    centroids = np.full((n_clusters, n_features), np.nan)
    centroid_sums = np.full((n_clusters, n_features), np.nan)
    centroid_counts = np.full(n_clusters, 0.0)
    return sample_membership, centroids, centroid_sums, centroid_counts


@njit
def sub_sections(n_samples, n_sections):
    n_samples = int(abs(n_samples)) # Распространить этот подход на другие функции
    n_sections = int(abs(n_sections))
    samples_per_section, n_extras = divmod(n_samples, n_sections)    
    if samples_per_section == 0:
        n_sections = n_extras        
    points = np.full(n_sections, samples_per_section)
    for i in range(n_extras):
        points[i] += 1        
    cumsum = 0
    for i in range(n_sections):
        cumsum += points[i]
        points[i] = cumsum        
    sections = np.empty((n_sections,2), dtype = points.dtype)    
    start_ind = 0
    for i in range(n_sections):
        sections[i,0] = start_ind
        sections[i,1] = points[i]
        start_ind = points[i]                
    return sections


def _assignment(samples, sample_membership, centroids):
    n_samples, n_features = samples.shape
    n_centroids = centroids.shape[0]
    n_sample_membership = sample_membership.shape[0]
    objective = 0.0
    n_changed_membership = 0
    for i in prange(n_samples):
        min_dist2 = np.inf
        min_ind = -1
        for j in range(n_centroids):
            if not np.isnan(centroids[j,0]):
                dist2 = 0.0
                for h in range(n_features):
                    dist2 += (centroids[j,h] - samples[i,h])**2
                if dist2 < min_dist2:
                    min_dist2 = dist2
                    min_ind = j
        if min_ind == -1: min_dist2 = np.nan                            
        if (n_sample_membership > 0) and (sample_membership[i] != min_ind):
            n_changed_membership += 1
            sample_membership[i] = min_ind
        objective += min_dist2
    return objective, n_changed_membership


assignment = njit(parallel=False)(_assignment)
assignment_parallel = njit(parallel=True)(_assignment)


@njit(parallel = False)
def update_centroids(samples, sample_membership, centroids, centroid_sums, centroid_counts):
    n_samples, n_features = samples.shape
    n_clusters = centroids.shape[0]   
    for i in range(n_clusters):
        centroid_counts[i] = 0.0
        for j in range(n_features):
            centroid_sums[i,j] = 0.0
            centroids[i,j] = np.nan
    for i in range(n_samples):
        centroid_ind = sample_membership[i]
        for j in range(n_features):
            centroid_sums[centroid_ind,j] += samples[i,j]
        centroid_counts[centroid_ind] += 1.0
    for i in range(n_clusters):
        if centroid_counts[i] > 0.0:
            for j in range(n_features):
                centroids[i,j] = centroid_sums[i,j] / centroid_counts[i]
                                

@njit(parallel = True)
def update_centroids_parallel(samples, sample_membership, centroids, centroid_sums, centroid_counts):
    n_samples, n_features = samples.shape
    n_clusters = centroids.shape[0]
    for i in range(n_clusters):
        centroid_counts[i] = 0.0
        for j in range(n_features):
            centroid_sums[i,j] = 0.0    
    thread_ranges = sub_sections(n_samples, nb.config.NUMBA_NUM_THREADS)
    n_threads = thread_ranges.shape[0]
    thread_centroid_sums = np.zeros((n_threads,n_clusters,n_features))
    thread_centroid_counts = np.zeros((n_threads,n_clusters))    
    for i in prange(n_threads):
        for j in range(thread_ranges[i,0],thread_ranges[i,1]):
            centroid_ind = sample_membership[j]
            for k in range(n_features):
                thread_centroid_sums[i,centroid_ind,k] += samples[j,k]
            thread_centroid_counts[i,centroid_ind] += 1.0        
    for i in range(n_threads):
        for j in range(n_clusters):
            centroid_counts[j] += thread_centroid_counts[i,j]
            for k in range(n_features):
                centroid_sums[j,k] += thread_centroid_sums[i,j,k]
    for i in range(n_clusters):
        if centroid_counts[i] > 0.0:
            for j in range(n_features):
                centroids[i,j] = centroid_sums[i,j] / centroid_counts[i]
        else:
            for j in range(n_features):
                centroids[i,j] = np.nan
                centroid_sums[i,j] = np.nan


@njit
def k_means(samples, sample_membership, centroids, centroid_sums, centroid_counts, max_iters = 300, tol=0.0001, parallel = True):
    n_samples, n_features = samples.shape
    n_clusters = centroids.shape[0]
    check_shapes(samples, sample_membership, centroids, centroid_sums, centroid_counts)
    objective = np.inf
    n_iters = 0
    sample_membership.fill(-1)
    if (n_samples > 0) and (n_features > 0) and (n_clusters > 0):
        n_changed_membership = 1
        objective_previous = np.inf
        tolerance = np.inf
        while True:
            if parallel:
                objective, n_changed_membership = assignment_parallel(samples, sample_membership, centroids)
            else:
                objective, n_changed_membership = assignment(samples, sample_membership, centroids)
            tolerance = 1 - objective/objective_previous
            objective_previous = objective
            n_iters += 1
            if (n_iters >= max_iters) or (n_changed_membership <= 0) or (tolerance <= tol) or (objective <= 0.0):
                break
            if parallel:
                update_centroids_parallel(samples, sample_membership, centroids, centroid_sums, centroid_counts)
            else:
                update_centroids(samples, sample_membership, centroids, centroid_sums, centroid_counts)
    return objective, n_iters


# Big-means with parallelization scheme #1: separate data portions are clustered sequentially one-by-one, but the clustering process itself is parallelized on the level of implementation of the K-means and K-means++ functions;
@njit
def big_means_par1(X, k = 3, s = 100, max_iter = 10000, tmax = 10, init_mode = 1, local_max_iters=300, local_tol=0.0001, n_candidates = 3, parallel = True, printing=False):
    m, n = X.shape
    assert s <= m
    if printing:
        with objmode:
            print ('%-30s%-15s%-15s' % ('objective', 'n_iter', 'cpu_time'))
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
    cpu_time = 0.0           
    membership = np.full(s, -1)
    centsums = np.empty((k, n))
    centnums = np.zeros(k)    
    centroids = np.full((k, n), np.nan)
    objective = np.inf
    n_dists = 0
    n_iter = 0
    best_time = 0.0
    best_n_dists = 0
    best_n_iter = 0       
    while (n_iter < max_iter) and (cpu_time < tmax):
        sample = X[np.random.choice(m, s, replace=False)]
        centers = np.copy(centroids)        
        degenerate_mask = np.sum(np.isnan(centers), axis = 1) > 0
        n_degenerate = np.sum(degenerate_mask)            
        if n_degenerate > 0:
            if init_mode == 0:
                new_centers = np.random.choice(s, n_degenerate, replace=False)
            else:
                if parallel:
                    new_centers, n_d = kmeanspp_parallel(sample, centers, n_degenerate, n_candidates)
                else:
                    new_centers, n_d = kmeanspp(sample, centers, n_degenerate, n_candidates)
                n_dists += n_d
            centers[degenerate_mask,:] = sample[new_centers,:]
        obj, n_it = k_means(sample, membership, centers, centsums, centnums, local_max_iters, local_tol, parallel)
        n_dists += n_it*s*k            
        with objmode(cpu_time = 'float64'):
            cpu_time = time.perf_counter() - start_time        
        n_iter += 1
        if obj < objective:
            objective = obj
            centroids = np.copy(centers)
            if printing:
                with objmode:
                    print ('%-30f%-15i%-15.2f' % (objective, n_iter, cpu_time))
            best_time = cpu_time
            best_n_dists = n_dists
            best_n_iter = n_iter    
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()        
    entity_membership = np.full(m, -1)
    if parallel:
        full_objective, n_changed_membership = assignment_parallel(X, entity_membership, centroids)
    else:
        full_objective, n_changed_membership = assignment(X, entity_membership, centroids)
    best_n_dists += k * m    
    with objmode(cpu_time = 'float64'):
        cpu_time = time.perf_counter() - start_time        
    best_time += cpu_time                        
    return centroids, full_objective, entity_membership, n_iter, best_n_iter, best_time, best_n_dists
                     

# Big-means with parallelization scheme #2: separate data portions are processed in parallel while each portion is clustered on a separate CPU core using the regular/sequential implementations of the K-means and K-means++ algorithms
@njit(parallel = True)
def big_means_par2(X, k = 3, sample_size = 100, n_samples = 1000, init_mode = 1, local_max_iters=300, local_tol=0.0001, n_candidates = 3, printing=False):
    m, n = X.shape
    assert sample_size <= m
    n_dists = np.full(n_samples, 0)
    solutions = np.full((n_samples, k, n), np.nan)
    objectives = np.full(n_samples, np.inf)
    for i in prange(n_samples):
        membership = np.full(sample_size, -1)
        centsums = np.empty((k, n))
        centnums = np.zeros(k)
        sample = X[np.random.choice(m, sample_size, replace=False)]
        best = np.argmin(objectives)
        objective = objectives[best]
        solution = np.copy(solutions[best])
        degenerate_mask = np.sum(np.isnan(solution), axis = 1) > 0
        n_degenerate = np.sum(degenerate_mask)
        if n_degenerate > 0:
            if init_mode == 0:
                new_centers = np.random.choice(sample_size, n_degenerate, replace=False)
            else:
                new_centers, n_d = kmeanspp(sample, solution, n_degenerate, n_candidates)
                n_dists[i] += n_d
            solution[degenerate_mask,:] = sample[new_centers,:]
        obj, n_it = k_means(sample, membership, solution, centsums, centnums, local_max_iters, local_tol, False)
        n_dists[i] += n_it*sample_size*k
        if obj < objective:
            objectives[i] = obj
            solutions[i] = np.copy(solution)
            if printing:
                print(objective)
    best = np.argmin(objectives)
    n_distances = np.sum(n_dists)    
    entity_membership = np.full(m, -1)
    final_objective, n_ch = assignment_parallel(X, entity_membership, solutions[best])
    n_distances += k * m    
    return solutions[best], final_objective, entity_membership, n_distances
