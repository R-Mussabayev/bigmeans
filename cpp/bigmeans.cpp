#ifndef _OPENMP
#error "This project requires OpenMP. Please compile with -fopenmp (GCC/Clang) or /openmp (MSVC)."
#endif

#include "bigmeans.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>

namespace bigmeans {
namespace {
    // Not exposed publicly
    inline Eigen::MatrixXd gather_rows(const Eigen::MatrixXd& A, const Eigen::VectorXi& idx) {
        Eigen::MatrixXd out(idx.size(), A.cols());
        for (int i = 0; i < idx.size(); ++i) out.row(i) = A.row(idx[i]);
        return out;
    }
} // anonymous


Eigen::MatrixXd distance_mat(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    // Compute dot product matrix between X and Y
    Eigen::MatrixXd out = X * Y.transpose(); // Equivalent to np.dot(X, Y.T)

    // Compute squared norms of rows of X and Y
    Eigen::VectorXd NX = X.rowwise().squaredNorm(); // Shape: (X_rows,)
    Eigen::VectorXd NY = Y.rowwise().squaredNorm(); // Shape: (Y_rows,)

    // Apply the distance formula: NX[i] - 2*out[i,j] + NY[j]
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < Y.rows(); ++j) {
            out(i, j) = NX(i) - 2.0 * out(i, j) + NY(j);
        }
    }

    return out;
}

    
Eigen::MatrixXd distance_mat_parallel(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    const int X_rows = X.rows();
    const int Y_rows = Y.rows();
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(X_rows, Y_rows);

    const Eigen::VectorXd NX = X.rowwise().squaredNorm();
    Eigen::VectorXd NY(Y_rows);
    for (int i = 0; i < Y_rows; ++i) NY(i) = Y.row(i).squaredNorm();

    #pragma omp parallel for
    for (int i = 0; i < Y_rows; ++i) {
        for (int j = 0; j < X_rows; ++j) {
            const double dot = X.row(j).dot(Y.row(i));
            out(j, i) = NX(j) - 2.0 * dot + NY(i);
        }
    }
    return out;
}


// Serial k-means++ (no OpenMP inside)
// Returns: (chosen_center_indices, n_dists_proxy)
std::pair<Eigen::VectorXi, std::uint64_t>
kmeans_plus_plus(const Eigen::MatrixXd& points,
                 const Eigen::MatrixXd& centers,
                 int n_new_centers,
                 int n_candidates,
                 std::mt19937_64* rng_ptr)
{
    const int n_points   = static_cast<int>(points.rows());
    const int n_features = static_cast<int>(points.cols());
    const int n_centers  = static_cast<int>(centers.rows());

    std::uint64_t n_dists = 0;
    Eigen::VectorXi center_inds = Eigen::VectorXi::Constant(n_new_centers, -1);

    // Guard conditions (same as Python)
    if (!(n_points > 0 && n_features > 0 && n_new_centers > 1 && n_candidates > 0)) {
        return {center_inds, n_dists};
    }

    // RNG (deterministic default)
    std::mt19937_64 local_rng(123456789ULL);
    std::mt19937_64& rng = rng_ptr ? *rng_ptr : local_rng;

    Eigen::VectorXd closest_dist_sq;  // size n_points
    int n_added_centers = 0;

    if (n_centers == 0) {
        // First center uniformly at random
        std::uniform_int_distribution<int> uid(0, n_points - 1);
        const int first = uid(rng);
        center_inds(0) = first;

        // distances from the first chosen point to all points (serial version)
        const Eigen::MatrixXd d = distance_mat(points.row(first), points);
        // distance_mat returns (1 x n_points); take row 0
        closest_dist_sq = d.row(0).transpose();
        n_dists += static_cast<std::uint64_t>(n_points);
        n_added_centers = 1;
    } else {
        // Start from existing centers (serial distance_mat)
        const Eigen::MatrixXd dist_mat = distance_mat(centers, points); // (n_centers x n_points)
        n_dists += static_cast<std::uint64_t>(n_centers) * static_cast<std::uint64_t>(n_points);

        // For each point, take min distance to any current center (serial)
        closest_dist_sq.resize(n_points);
        for (int j = 0; j < n_points; ++j) {
            double m = dist_mat(0, j);
            for (int i = 1; i < n_centers; ++i) {
                const double v = dist_mat(i, j);
                if (v < m) m = v;
            }
            closest_dist_sq(j) = m;
        }
        n_added_centers = 0;
    }

    double current_pot = closest_dist_sq.sum();

    // Add remaining new centers
    for (int c = n_added_centers; c < n_new_centers; ++c) {
        // Prefix-sum of closest_dist_sq
        Eigen::VectorXd prefix = closest_dist_sq;
        for (int i = 1; i < n_points; ++i) prefix(i) += prefix(i - 1);

        // Draw n_candidates samples proportional to closest_dist_sq
        std::uniform_real_distribution<double> urand(0.0, current_pot);
        Eigen::VectorXi candidate_ids(n_candidates);
        for (int t = 0; t < n_candidates; ++t) {
            const double r = urand(rng);
            const double* beg = prefix.data();
            const double* end = beg + n_points;
            const double* it  = std::upper_bound(beg, end, r);
            int idx = static_cast<int>(std::distance(beg, it));
            if (idx >= n_points) idx = n_points - 1; // guard
            candidate_ids(t) = idx;
        }

        // Distances from each candidate to all points (serial distance_mat)
        const Eigen::MatrixXd cand_pts = gather_rows(points, candidate_ids);
        Eigen::MatrixXd dists = distance_mat(cand_pts, points); // (n_candidates x n_points)
        n_dists += static_cast<std::uint64_t>(dists.size());

        // Elementwise min with current closest_dist_sq and compute pot per candidate
        const int ncols = dists.cols();
        Eigen::VectorXd candidates_pot(n_candidates);
        for (int r = 0; r < n_candidates; ++r) {
            double s = 0.0;
            for (int j = 0; j < ncols; ++j) {
                double v = dists(r, j);
                const double w = closest_dist_sq(j);
                if (w < v) v = w;
                dists(r, j) = v; // store min for the "commit" step if chosen
                s += v;
            }
            candidates_pot(r) = s;
        }

        // Best candidate = argmin potential
        Eigen::Index best_r;
        candidates_pot.minCoeff(&best_r);

        // Commit: update potential, per-point closest distances, and center index
        current_pot     = candidates_pot(best_r);
        closest_dist_sq = dists.row(static_cast<int>(best_r)).transpose();
        center_inds(c)  = candidate_ids(static_cast<int>(best_r));
    }

    return {center_inds, n_dists};
}

    
std::pair<Eigen::VectorXi, std::uint64_t>
kmeans_plus_plus_parallel(const Eigen::MatrixXd& points,
                          const Eigen::MatrixXd& centers,
                          int n_new_centers,
                          int n_candidates,
                          std::mt19937_64* rng_ptr)
{
    const int n_points   = static_cast<int>(points.rows());
    const int n_features = static_cast<int>(points.cols());
    const int n_centers  = static_cast<int>(centers.rows());

    std::uint64_t n_dists = 0;
    Eigen::VectorXi center_inds = Eigen::VectorXi::Constant(n_new_centers, -1);

    if (!(n_points > 0 && n_features > 0 && n_new_centers > 1 && n_candidates > 0)) {
        return {center_inds, n_dists};
    }

    std::mt19937_64 local_rng(123456789ULL);
    std::mt19937_64& rng = rng_ptr ? *rng_ptr : local_rng;

    Eigen::VectorXd closest_dist_sq;
    int n_added_centers = 0;

    if (n_centers == 0) {
        std::uniform_int_distribution<int> uid(0, n_points - 1);
        const int first = uid(rng);
        center_inds(0) = first;

        const Eigen::MatrixXd d = distance_mat_parallel(points.row(first), points);
        closest_dist_sq = d.row(0).transpose();
        n_dists += static_cast<std::uint64_t>(n_points);
        n_added_centers = 1;
    } else {
        const Eigen::MatrixXd dist_mat = distance_mat_parallel(centers, points);
        n_dists += static_cast<std::uint64_t>(n_centers) * static_cast<std::uint64_t>(n_points);

        closest_dist_sq.resize(n_points);
        #pragma omp parallel for
        for (int j = 0; j < n_points; ++j) {
            double m = dist_mat(0, j);
            for (int i = 1; i < n_centers; ++i) if (dist_mat(i, j) < m) m = dist_mat(i, j);
            closest_dist_sq(j) = m;
        }
    }

    double current_pot = closest_dist_sq.sum();

    for (int c = n_added_centers; c < n_new_centers; ++c) {
        Eigen::VectorXd prefix = closest_dist_sq;
        for (int i = 1; i < n_points; ++i) prefix(i) += prefix(i - 1);

        std::uniform_real_distribution<double> urand(0.0, current_pot);
        Eigen::VectorXi candidate_ids(n_candidates);
        for (int t = 0; t < n_candidates; ++t) {
            const double r = urand(rng);
            const double* beg = prefix.data();
            const double* end = beg + n_points;
            const double* it  = std::upper_bound(beg, end, r);
            int idx = static_cast<int>(std::distance(beg, it));
            if (idx >= n_points) idx = n_points - 1;
            candidate_ids(t) = idx;
        }

        const Eigen::MatrixXd cand_pts = gather_rows(points, candidate_ids);
        Eigen::MatrixXd dists = distance_mat_parallel(cand_pts, points);
        n_dists += static_cast<std::uint64_t>(dists.size());

        Eigen::VectorXd candidates_pot(n_candidates);
        const int ncols = dists.cols();

        #pragma omp parallel for
        for (int r = 0; r < n_candidates; ++r) {
            double s = 0.0;
            for (int j = 0; j < ncols; ++j) {
                double v = dists(r, j);
                if (closest_dist_sq(j) < v) v = closest_dist_sq(j);
                dists(r, j) = v;
                s += v;
            }
            candidates_pot(r) = s;
        }

        Eigen::Index best_r;
        candidates_pot.minCoeff(&best_r);

        current_pot       = candidates_pot(best_r);
        closest_dist_sq   = dists.row(static_cast<int>(best_r)).transpose();
        center_inds(c)    = candidate_ids(static_cast<int>(best_r));
    }

    return {center_inds, n_dists};
}


std::tuple<double, int, Eigen::VectorXi, std::uint64_t>
kmeans(const Eigen::MatrixXd& points,
       Eigen::MatrixXd&       centers,
       int                    max_iters,
       double                 tol)
{
    const int m = static_cast<int>(points.rows());
    const int n = static_cast<int>(points.cols());
    const int k = static_cast<int>(centers.rows());

    Eigen::VectorXi assignment = Eigen::VectorXi::Constant(m, -1);
    Eigen::MatrixXd center_sums(k, n);
    Eigen::VectorXd center_counts(k);

    // Precompute point norms once
    const Eigen::VectorXd NX = points.rowwise().squaredNorm();

    double f = std::numeric_limits<double>::infinity();
    int n_iters = 0;

    if (m > 0 && n > 0 && k > 0) {
        double prev_obj = std::numeric_limits<double>::infinity();

        while (true) {
            // ---- Assignment step (serial) ----
            const Eigen::VectorXd NC = centers.rowwise().squaredNorm();

            double f_acc = 0.0;   // accumulate objective directly
            int n_changed = 0;    // count changes directly

            for (int i = 0; i < m; ++i) {
                const double nx = NX(i);

                // Start with center 0
                double best_d = nx + NC(0) - 2.0 * points.row(i).dot(centers.row(0));
                int    best_j = 0;

                // Scan other centers
                for (int j = 1; j < k; ++j) {
                    const double d = nx + NC(j) - 2.0 * points.row(i).dot(centers.row(j));
                    if (d < best_d) { best_d = d; best_j = j; }
                }

                // Accumulate objective
                f_acc += best_d;

                // Count change and update assignment in-place
                if (assignment(i) != best_j) ++n_changed;
                assignment(i) = best_j;
            }

            f = f_acc;
            ++n_iters;

            const double improvement = (prev_obj < std::numeric_limits<double>::infinity())
                                       ? (1.0 - f / prev_obj)
                                       : std::numeric_limits<double>::infinity();
            prev_obj = f;

            if ((max_iters >= 0 && n_iters >= max_iters) ||
                (n_changed == 0) ||
                (tol > 0.0 && improvement <= tol)) {
                break;
            }

            // ---- Update step (serial) ----
            center_sums.setZero();
            center_counts.setZero();

            for (int i = 0; i < m; ++i) {
                const int ci = assignment(i);
                if (ci >= 0) {
                    center_sums.row(ci) += points.row(i);
                    center_counts(ci)   += 1.0;
                }
            }

            for (int j = 0; j < k; ++j) {
                const double cnt = center_counts(j);
                if (cnt > 0.0) {
                    centers.row(j) = center_sums.row(j) / cnt;
                }
            }
        }
    }

    const std::uint64_t n_ops = static_cast<std::uint64_t>(n_iters)
                              * static_cast<std::uint64_t>(k)
                              * static_cast<std::uint64_t>(m);
    return {f, n_iters, assignment, n_ops};
}

    
std::tuple<double, int, Eigen::VectorXi, std::uint64_t>
kmeans_parallel(const Eigen::MatrixXd& points,
                Eigen::MatrixXd&       centers,
                int                    max_iters,
                double                 tol)
{
    const int m = static_cast<int>(points.rows());
    const int n = static_cast<int>(points.cols());
    const int k = static_cast<int>(centers.rows());

    Eigen::VectorXi assignment = Eigen::VectorXi::Constant(m, -1);
    Eigen::MatrixXd center_sums(k, n);
    Eigen::VectorXd center_counts(k);
    Eigen::VectorXi new_assignment(m);
    Eigen::VectorXd min_dists(m);
    Eigen::VectorXi changed_flags(m);

    const Eigen::VectorXd NX = points.rowwise().squaredNorm();

    double f = std::numeric_limits<double>::infinity();
    int n_iters = 0;

    if (m > 0 && n > 0 && k > 0) {
        double prev_obj = std::numeric_limits<double>::infinity();

        while (true) {
            const Eigen::VectorXd NC = centers.rowwise().squaredNorm();

            #pragma omp parallel for
            for (int i = 0; i < m; ++i) {
                const double nx = NX(i);
                double best_d = nx + NC(0) - 2.0 * points.row(i).dot(centers.row(0));
                int    best_j = 0;
                for (int j = 1; j < k; ++j) {
                    const double d = nx + NC(j) - 2.0 * points.row(i).dot(centers.row(j));
                    if (d < best_d) { best_d = d; best_j = j; }
                }
                new_assignment(i) = best_j;
                min_dists(i)      = best_d;
                changed_flags(i)  = (assignment(i) != best_j) ? 1 : 0;
            }

            f = 0.0;
            int n_changed = 0;
            for (int i = 0; i < m; ++i) {
                f += min_dists(i);
                n_changed += changed_flags(i);
                assignment(i) = new_assignment(i);
            }

            ++n_iters;
            const double improvement = (prev_obj < std::numeric_limits<double>::infinity())
                                     ? (1.0 - f / prev_obj)
                                     : std::numeric_limits<double>::infinity();
            prev_obj = f;

            if ((max_iters >= 0 && n_iters >= max_iters) ||
                (n_changed == 0) ||
                (tol > 0.0 && improvement <= tol)) {
                break;
            }

            center_sums.setZero();
            center_counts.setZero();
            for (int i = 0; i < m; ++i) {
                const int ci = assignment(i);
                if (ci >= 0) {
                    center_sums.row(ci) += points.row(i);
                    center_counts(ci)   += 1.0;
                }
            }
            for (int j = 0; j < k; ++j) {
                const double cnt = center_counts(j);
                if (cnt > 0.0) centers.row(j) = center_sums.row(j) / cnt;
            }
        }
    }

    const std::uint64_t n_ops = static_cast<std::uint64_t>(n_iters)
                              * static_cast<std::uint64_t>(k)
                              * static_cast<std::uint64_t>(m);
    return {f, n_iters, assignment, n_ops};
}


// Fully Sequential Big-means
// Returns: (centers, full_objective, assignment, n_iter, best_n_iter, best_time, n_dists)
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_sequential(const Eigen::MatrixXd& points,
                     int   n_centers,
                     int   sample_size,
                     int   max_iter,
                     double tmax,
                     int   local_max_iters,
                     double local_tol,
                     int   n_candidates,
                     bool  printing,
                     std::mt19937_64* rng_ptr)
{
    using clock = std::chrono::steady_clock;

    const int n_points   = static_cast<int>(points.rows());
    const int n_features = static_cast<int>(points.cols());
    if (!(sample_size <= n_points) || n_points == 0 || n_features == 0 || n_centers <= 0) {
        Eigen::MatrixXd empty_centers = Eigen::MatrixXd::Constant(
            std::max(0, n_centers), std::max(0, n_features),
            std::numeric_limits<double>::quiet_NaN());
        return {empty_centers, std::numeric_limits<double>::infinity(),
                Eigen::VectorXi(), 0, 0, 0.0, 0};
    }

    // RNG (deterministic default)
    std::mt19937_64 local_rng(123456789ULL);
    std::mt19937_64& rng = rng_ptr ? *rng_ptr : local_rng;

    // Timer start
    const auto t0 = clock::now();
    auto seconds_since_start = [&](const clock::time_point& now) -> double {
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - t0).count();
    };

    // State
    Eigen::MatrixXd centers = Eigen::MatrixXd::Constant(
        n_centers, n_features, std::numeric_limits<double>::quiet_NaN());
    double        objective    = std::numeric_limits<double>::infinity();
    int           n_iter       = 0;
    double        best_time    = 0.0;
    int           best_n_iter  = 0;
    std::uint64_t n_dists      = 0;

    if (printing) {
        std::cout << std::left << std::setw(30) << "sample objective"
                  << std::setw(15) << "n_iter"
                  << std::setw(15) << "cpu_time" << "\n";
    }

    // Sample k indices without replacement
    auto sample_indices = [&](int N, int k) -> Eigen::VectorXi {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        for (int i = 0; i < k; ++i) {
            std::uniform_int_distribution<int> dist(i, N - 1);
            const int j = dist(rng);
            std::swap(idx[i], idx[j]);
        }
        Eigen::VectorXi out(k);
        for (int i = 0; i < k; ++i) out(i) = idx[i];
        return out;
    };

    auto is_row_nan = [&](const Eigen::MatrixXd& M, int r) -> bool {
        for (int c = 0; c < M.cols(); ++c) if (std::isnan(M(r, c))) return true;
        return false;
    };

    // Main loop: iterate over random samples until limits
    while ((max_iter <= 0 || n_iter < max_iter)) {
        const double elapsed = seconds_since_start(clock::now());
        if (tmax > 0.0 && elapsed >= tmax) break;

        const Eigen::VectorXi sample_idx = sample_indices(n_points, sample_size);
        const Eigen::MatrixXd sample     = gather_rows(points, sample_idx);

        Eigen::MatrixXd new_centers = centers;

        // Identify degenerate centers (rows with NaN) and fill them via serial k-means++
        std::vector<int> deg_idx, good_idx;
        deg_idx.reserve(n_centers);
        good_idx.reserve(n_centers);
        for (int r = 0; r < n_centers; ++r) {
            (is_row_nan(new_centers, r) ? deg_idx : good_idx).push_back(r);
        }
        const int n_degenerate = static_cast<int>(deg_idx.size());

        if (n_degenerate > 0) {
            Eigen::MatrixXd existing_centers;
            if (!good_idx.empty()) {
                Eigen::VectorXi g(static_cast<int>(good_idx.size()));
                for (int i = 0; i < g.size(); ++i) g(i) = good_idx[i];
                existing_centers = gather_rows(new_centers, g);
            } else {
                existing_centers.resize(0, n_features);
            }

            // Serial k-means++ (not the parallel variant)
            Eigen::VectorXi center_inds_pp;
            std::uint64_t   num_dists_pp = 0;
            std::tie(center_inds_pp, num_dists_pp) =
                kmeans_plus_plus(sample, existing_centers, n_degenerate, n_candidates, &rng);
            n_dists += num_dists_pp;

            for (int t = 0; t < n_degenerate; ++t) {
                const int cr = deg_idx[t];
                const int sr = center_inds_pp(t);
                new_centers.row(cr) = sample.row(sr);
            }
        }

        // Serial k-means on the sample
        double new_objective = 0.0;
        int iters_unused = 0;
        Eigen::VectorXi assignment_unused;
        std::uint64_t num_dists_km = 0;
        std::tie(new_objective, iters_unused, assignment_unused, num_dists_km) =
            kmeans(sample, new_centers, local_max_iters, local_tol);
        n_dists += num_dists_km;

        const double cpu_time = seconds_since_start(clock::now());
        ++n_iter;

        if (new_objective < objective) {
            objective = new_objective;
            centers   = new_centers;
            if (printing) {
                std::cout << std::left << std::setw(30) << std::setprecision(6) << std::fixed << objective
                          << std::setw(15) << n_iter
                          << std::setw(15) << std::setprecision(2) << std::fixed << cpu_time
                          << "\n";
            }
            best_time   = cpu_time;
            best_n_iter = n_iter;
        }
    }

    // Final assignment-only pass (max_iters=0) using serial k-means
    double full_objective = 0.0;
    int iters_full = 0;
    Eigen::VectorXi assignment;
    std::uint64_t num_dists_full = 0;
    std::tie(full_objective, iters_full, assignment, num_dists_full) =
        kmeans(points, centers, 0, local_tol);
    n_dists += num_dists_full;

    return {centers, full_objective, assignment, n_iter, best_n_iter, best_time, n_dists};
}

    

// Big-means with "Inner Parallelism":
// Separate data samples are clustered sequentially one-by-one, but the clustering process itself 
// is parallelized on the level of internal implementation of the K-means and K-means++ functions.    
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_inner(const Eigen::MatrixXd& points,
                int   n_centers,
                int   sample_size,
                int   max_iter,
                double tmax,
                int   local_max_iters,
                double local_tol,
                int   n_candidates,
                bool  printing,
                std::mt19937_64* rng_ptr)
{
    using clock = std::chrono::steady_clock;

    const int n_points   = static_cast<int>(points.rows());
    const int n_features = static_cast<int>(points.cols());
    if (!(sample_size <= n_points) || n_points == 0 || n_features == 0 || n_centers <= 0) {
        Eigen::MatrixXd empty_centers = Eigen::MatrixXd::Constant(
            std::max(0, n_centers), std::max(0, n_features),
            std::numeric_limits<double>::quiet_NaN());
        return {empty_centers, std::numeric_limits<double>::infinity(),
                Eigen::VectorXi(), 0, 0, 0.0, 0};
    }

    std::mt19937_64 local_rng(123456789ULL);
    std::mt19937_64& rng = rng_ptr ? *rng_ptr : local_rng;

    const auto t0 = clock::now();
    auto seconds_since_start = [&](const clock::time_point& now) -> double {
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - t0).count();
    };

    Eigen::MatrixXd centers = Eigen::MatrixXd::Constant(
        n_centers, n_features, std::numeric_limits<double>::quiet_NaN());
    double  objective    = std::numeric_limits<double>::infinity();
    int     n_iter       = 0;
    double  best_time    = 0.0;
    int     best_n_iter  = 0;
    std::uint64_t n_dists = 0;

    if (printing) {
        std::cout << std::left << std::setw(30) << "sample objective"
                  << std::setw(15) << "n_iter"
                  << std::setw(15) << "cpu_time" << "\n";
    }

    auto sample_indices = [&](int N, int k) -> Eigen::VectorXi {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        for (int i = 0; i < k; ++i) {
            std::uniform_int_distribution<int> dist(i, N - 1);
            const int j = dist(rng);
            std::swap(idx[i], idx[j]);
        }
        Eigen::VectorXi out(k);
        for (int i = 0; i < k; ++i) out(i) = idx[i];
        return out;
    };

    auto is_row_nan = [&](const Eigen::MatrixXd& M, int r) -> bool {
        for (int c = 0; c < M.cols(); ++c) if (std::isnan(M(r, c))) return true;
        return false;
    };

    while ((max_iter <= 0 || n_iter < max_iter)) {
        const double elapsed = seconds_since_start(clock::now());
        if (tmax > 0.0 && elapsed >= tmax) break;

        const Eigen::VectorXi sample_idx = sample_indices(n_points, sample_size);
        const Eigen::MatrixXd sample     = gather_rows(points, sample_idx);

        Eigen::MatrixXd new_centers = centers;

        std::vector<int> deg_idx, good_idx;
        deg_idx.reserve(n_centers);
        good_idx.reserve(n_centers);
        for (int r = 0; r < n_centers; ++r) {
            (is_row_nan(new_centers, r) ? deg_idx : good_idx).push_back(r);
        }
        const int n_degenerate = static_cast<int>(deg_idx.size());

        if (n_degenerate > 0) {
            Eigen::MatrixXd existing_centers;
            if (!good_idx.empty()) {
                Eigen::VectorXi g(static_cast<int>(good_idx.size()));
                for (int i = 0; i < g.size(); ++i) g(i) = good_idx[i];
                existing_centers = gather_rows(new_centers, g);
            } else {
                existing_centers.resize(0, n_features);
            }

            auto pp = kmeans_plus_plus_parallel(sample, existing_centers,
                                                n_degenerate, n_candidates, &rng);
            Eigen::VectorXi center_inds_pp;
            std::uint64_t   num_dists_pp = 0;
            std::tie(center_inds_pp, num_dists_pp) = pp;
            n_dists += num_dists_pp;

            for (int t = 0; t < n_degenerate; ++t) {
                const int cr = deg_idx[t];
                const int sr = center_inds_pp(t);
                new_centers.row(cr) = sample.row(sr);
            }
        }

        double new_objective = 0.0;
        int iters_unused = 0;
        Eigen::VectorXi assignment_unused;
        std::uint64_t num_dists_km = 0;
        std::tie(new_objective, iters_unused, assignment_unused, num_dists_km) =
            kmeans_parallel(sample, new_centers, local_max_iters, local_tol);
        n_dists += num_dists_km;

        const double cpu_time = seconds_since_start(clock::now());
        ++n_iter;

        if (new_objective < objective) {
            objective = new_objective;
            centers   = new_centers;
            if (printing) {
                std::cout << std::left << std::setw(30) << std::setprecision(6) << std::fixed << objective
                          << std::setw(15) << n_iter
                          << std::setw(15) << std::setprecision(2) << std::fixed << cpu_time
                          << "\n";
            }
            best_time   = cpu_time;
            best_n_iter = n_iter;
        }
    }

    // ----- Final assignment-only pass (max_iters=0) -----
    double full_objective = 0.0;
    int iters_full = 0;
    Eigen::VectorXi assignment;
    std::uint64_t num_dists_full = 0;
    std::tie(full_objective, iters_full, assignment, num_dists_full) =
        kmeans_parallel(points, centers, 0, 0.0);
    n_dists += num_dists_full;

    return {centers, full_objective, assignment, n_iter, best_n_iter, best_time, n_dists};
}



// Big-means with "Competitive Parallelism":
// Separate data samples are processed in parallel while each sample is clustered 
// on a separate CPU core using the regular / sequential implementations of 
// the K-means and K-means++ algorithms. Workers use only their previous own best centroids 
// for initialization at every iteration. This parallelization mode is called competitive since 
// all workers are independent and compete with each other.
// Returns: (centers, full_objective, assignment, total_iters, best_n_iter, best_time, total_dists)
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_competitive(const Eigen::MatrixXd& points,
                      int   n_centers,
                      int   sample_size,
                      int   max_iter,
                      double tmax,
                      int   local_max_iters,
                      double local_tol,
                      int   n_candidates,
                      bool  printing,
                      std::mt19937_64* rng_ptr)
{
    using clock = std::chrono::steady_clock;

    const int n_points   = static_cast<int>(points.rows());
    const int n_features = static_cast<int>(points.cols());
    if (!(sample_size <= n_points) || n_points == 0 || n_features == 0 || n_centers <= 0) {
        Eigen::MatrixXd empty_centers = Eigen::MatrixXd::Constant(
            std::max(0, n_centers), std::max(0, n_features),
            std::numeric_limits<double>::quiet_NaN());
        return {empty_centers, std::numeric_limits<double>::infinity(),
                Eigen::VectorXi(), 0, 0, 0.0, 0};
    }

    // Threads
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // Timer (shared start)
    const auto t0 = clock::now();
    auto seconds_since_start = [&](const clock::time_point& now) -> double {
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - t0).count();
    };

    // Per-thread state
    std::vector<Eigen::MatrixXd> centers_vec(n_threads, Eigen::MatrixXd::Constant(
        n_centers, n_features, std::numeric_limits<double>::quiet_NaN()));
    std::vector<double>        objectives(n_threads, std::numeric_limits<double>::infinity());
    std::vector<std::uint64_t> n_dists_vec(n_threads, 0);
    std::vector<int>           n_iters_vec(n_threads, 0);
    std::vector<double>        running_time(n_threads, 0.0);
    std::vector<double>        best_times(n_threads, 0.0);
    std::vector<int>           best_n_iters(n_threads, 0);

    // Global counters/best
    std::atomic<int> total_iters{0};
    double global_best = std::numeric_limits<double>::infinity();

    // Optional header
    if (printing) {
        std::cout << std::left << std::setw(30) << "sample objective"
                  << std::setw(15) << "n_iter"
                  << std::setw(15) << "cpu_time" << "\n";
    }

    // Helper: sample k indices without replacement
    auto sample_indices = [](int N, int k, std::mt19937_64& rng) -> Eigen::VectorXi {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        for (int i = 0; i < k; ++i) {
            std::uniform_int_distribution<int> dist(i, N - 1);
            const int j = dist(rng);
            std::swap(idx[i], idx[j]);
        }
        Eigen::VectorXi out(k);
        for (int i = 0; i < k; ++i) out(i) = idx[i];
        return out;
    };

    auto is_row_nan = [&](const Eigen::MatrixXd& M, int r) -> bool {
        for (int c = 0; c < M.cols(); ++c) if (std::isnan(M(r, c))) return true;
        return false;
    };

    // Base RNG seed (deterministic unless user passes rng_ptr)
    std::uint64_t base_seed = 123456789ULL;
    if (rng_ptr) {
        // derive base seed from user RNG without consuming too much state
        base_seed ^= (*rng_ptr)();
    }

    // Parallel region: each worker loops independently ("competitive")
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
#ifdef _OPENMP
        const int t = omp_get_thread_num();
#else
        const int t = 0;
#endif
        // Per-thread RNG
        std::mt19937_64 rng(base_seed + static_cast<std::uint64_t>(t) * 0x9E3779B97F4A7C15ULL);

        while ((max_iter <= 0 || total_iters.load(std::memory_order_relaxed) < max_iter)) {
            const double now = seconds_since_start(clock::now());
            if (tmax > 0.0 && now >= tmax) break;

            // Draw a sample
            const Eigen::VectorXi sample_idx = sample_indices(n_points, sample_size, rng);
            const Eigen::MatrixXd sample     = gather_rows(points, sample_idx);

            // Initialize new_centers from this thread's current best centers
            Eigen::MatrixXd new_centers = centers_vec[t];

            // Degenerate rows (NaN) -> fill using serial k-means++
            std::vector<int> deg_idx, good_idx;
            deg_idx.reserve(n_centers);
            good_idx.reserve(n_centers);
            for (int r = 0; r < n_centers; ++r) {
                (is_row_nan(new_centers, r) ? deg_idx : good_idx).push_back(r);
            }
            const int n_degenerate = static_cast<int>(deg_idx.size());

            if (n_degenerate > 0) {
                Eigen::MatrixXd existing_centers;
                if (!good_idx.empty()) {
                    Eigen::VectorXi g(static_cast<int>(good_idx.size()));
                    for (int i = 0; i < g.size(); ++i) g(i) = good_idx[i];
                    existing_centers = gather_rows(new_centers, g);
                } else {
                    existing_centers.resize(0, n_features);
                }

                Eigen::VectorXi center_inds_pp;
                std::uint64_t   num_dists_pp = 0;
                std::tie(center_inds_pp, num_dists_pp) =
                    kmeans_plus_plus(sample, existing_centers, n_degenerate, n_candidates, &rng);
                n_dists_vec[t] += num_dists_pp;

                for (int u = 0; u < n_degenerate; ++u) {
                    const int cr = deg_idx[u];
                    const int sr = center_inds_pp(u);
                    new_centers.row(cr) = sample.row(sr);
                }
            }

            // Serial k-means on the sample
            double new_objective = 0.0;
            int iters_unused = 0;
            Eigen::VectorXi assign_unused;
            std::uint64_t num_dists_km = 0;
            std::tie(new_objective, iters_unused, assign_unused, num_dists_km) =
                kmeans(sample, new_centers, local_max_iters, local_tol);
            n_dists_vec[t] += num_dists_km;

            // Update per-thread timing and iteration counters
            const double time_now = seconds_since_start(clock::now());
            running_time[t] = time_now;
            n_iters_vec[t] += 1;
            const int new_total_iters = total_iters.fetch_add(1, std::memory_order_relaxed) + 1;

            // If the thread improved its own best, record and maybe print as global best
            if (new_objective < objectives[t]) {
                objectives[t]  = new_objective;
                centers_vec[t] = new_centers;
                best_times[t]  = time_now;
                best_n_iters[t]= new_total_iters;

                bool is_global_best = false;
#ifdef _OPENMP
#pragma omp critical(bigmeans_globalbest)
#endif
                {
                    if (new_objective < global_best) {
                        global_best = new_objective;
                        is_global_best = true;
                    }
                }
                if (printing && is_global_best) {
                    std::cout << std::left << std::setw(30) << std::setprecision(6) << std::fixed << new_objective
                              << std::setw(15) << new_total_iters
                              << std::setw(15) << std::setprecision(2) << std::fixed << time_now
                              << "\n";
                }
            }

            // Check stopping conditions again (outer loop guards them already)
            if (max_iter > 0 && total_iters.load(std::memory_order_relaxed) >= max_iter) break;
            if (tmax > 0.0 && time_now >= tmax) break;
        }
    } // end parallel region

    // Pick globally best thread
    int best_ind = 0;
    for (int t = 1; t < n_threads; ++t) {
        if (objectives[t] < objectives[best_ind]) best_ind = t;
    }
    Eigen::MatrixXd final_centers = centers_vec[best_ind];

    // Final pass: assignment-only (max_iters=0) — use the parallel k-means for speed
    double full_objective = 0.0;
    int iters_full = 0;
    Eigen::VectorXi assignment;
    std::uint64_t num_dists_full = 0;
    std::tie(full_objective, iters_full, assignment, num_dists_full) =
        kmeans_parallel(points, final_centers, 0, 0.0);

    // Reductions
    std::uint64_t total_dists = num_dists_full;
    for (int t = 0; t < n_threads; ++t) total_dists += n_dists_vec[t];
    int total_iters_final = total_iters.load();

    return {final_centers, full_objective, assignment,
            total_iters_final, best_n_iters[best_ind], best_times[best_ind], total_dists};
}



// Big-means with "Collective Parallelism":
// Separate data samples are processed in parallel while each sample is clustered 
// on a separate CPU core using the regular / sequential implementations 
// of the K-means and K-means++ algorithms. At all subsequent iterations, 
// each worker uses the best set of centroids among all workers obtained
// at previous iterations to initialize a new random data sample.
// This parallelization mode is called collective since 
// the workers share information about the best solutions.
// Returns: (centers, full_objective, assignment, total_iters, best_n_iter, best_time, total_dists)
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_collective(const Eigen::MatrixXd& points,
                     int   n_centers,
                     int   sample_size,
                     int   max_iter,
                     double tmax,
                     int   local_max_iters,
                     double local_tol,
                     int   n_candidates,
                     bool  printing,
                     std::mt19937_64* rng_ptr)
{
    using clock = std::chrono::steady_clock;

    const int n_points   = static_cast<int>(points.rows());
    const int n_features = static_cast<int>(points.cols());
    if (!(sample_size <= n_points) || n_points == 0 || n_features == 0 || n_centers <= 0) {
        Eigen::MatrixXd empty_centers = Eigen::MatrixXd::Constant(
            std::max(0, n_centers), std::max(0, n_features),
            std::numeric_limits<double>::quiet_NaN());
        return {empty_centers, std::numeric_limits<double>::infinity(),
                Eigen::VectorXi(), 0, 0, 0.0, 0};
    }

    // Number of worker threads
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // Global start time
    const auto t0 = clock::now();
    auto seconds_since_start = [&](const clock::time_point& now) -> double {
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - t0).count();
    };

    // Per-thread state (each thread maintains its own best)
    std::vector<Eigen::MatrixXd> centers_vec(n_threads, Eigen::MatrixXd::Constant(
        n_centers, n_features, std::numeric_limits<double>::quiet_NaN()));
    std::vector<double>        objectives(n_threads, std::numeric_limits<double>::infinity());
    std::vector<std::uint64_t> n_dists_vec(n_threads, 0);
    std::vector<int>           n_iters_vec(n_threads, 0);
    std::vector<double>        running_time(n_threads, 0.0);
    std::vector<double>        best_times(n_threads, 0.0);
    std::vector<int>           best_n_iters(n_threads, 0);

    // Global iteration counter (sum over threads)
    std::atomic<int> total_iters{0};

    // Optional header
    if (printing) {
        std::cout << std::left << std::setw(30) << "sample objective"
                  << std::setw(15) << "n_iter"
                  << std::setw(15) << "cpu_time" << "\n";
    }

    // Helper: sample k indices without replacement
    auto sample_indices = [](int N, int k, std::mt19937_64& rng) -> Eigen::VectorXi {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        for (int i = 0; i < k; ++i) {
            std::uniform_int_distribution<int> dist(i, N - 1);
            const int j = dist(rng);
            std::swap(idx[i], idx[j]);
        }
        Eigen::VectorXi out(k);
        for (int i = 0; i < k; ++i) out(i) = idx[i];
        return out;
    };

    auto is_row_nan = [&](const Eigen::MatrixXd& M, int r) -> bool {
        for (int c = 0; c < M.cols(); ++c) if (std::isnan(M(r, c))) return true;
        return false;
    };

    // Seed base (deterministic, optionally derived from rng_ptr)
    std::uint64_t base_seed = 123456789ULL;
    if (rng_ptr) base_seed ^= (*rng_ptr)();

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
#ifdef _OPENMP
        const int t = omp_get_thread_num();
#else
        const int t = 0;
#endif
        // Per-thread RNG
        std::mt19937_64 rng(base_seed + static_cast<std::uint64_t>(t) * 0x9E3779B97F4A7C15ULL);

        while ((max_iter <= 0 || total_iters.load(std::memory_order_relaxed) < max_iter)) {
            const double now = seconds_since_start(clock::now());
            if (tmax > 0.0 && now >= tmax) break;

            // Take a snapshot of CURRENT global best
            int best_idx = 0;
            double best_obj = objectives[0];
            for (int u = 1; u < n_threads; ++u) {
                if (objectives[u] < best_obj) { best_obj = objectives[u]; best_idx = u; }
            }

            Eigen::MatrixXd init_centers = centers_vec[best_idx]; // may contain NaNs if not yet set

            // --- Get a random sample ---
            const Eigen::VectorXi sample_idx = sample_indices(n_points, sample_size, rng);
            const Eigen::MatrixXd sample     = gather_rows(points, sample_idx);

            // --- Initialize from snapshot-best centers (collective) ---
            Eigen::MatrixXd new_centers = init_centers;

            // Fill degenerate rows (NaNs) using serial k-means++
            std::vector<int> deg_idx, good_idx;
            deg_idx.reserve(n_centers);
            good_idx.reserve(n_centers);
            for (int r = 0; r < n_centers; ++r) {
                (is_row_nan(new_centers, r) ? deg_idx : good_idx).push_back(r);
            }
            const int n_degenerate = static_cast<int>(deg_idx.size());

            if (n_degenerate > 0) {
                Eigen::MatrixXd existing_centers;
                if (!good_idx.empty()) {
                    Eigen::VectorXi g(static_cast<int>(good_idx.size()));
                    for (int i = 0; i < g.size(); ++i) g(i) = good_idx[i];
                    existing_centers = gather_rows(new_centers, g);
                } else {
                    existing_centers.resize(0, n_features);
                }

                Eigen::VectorXi center_inds_pp;
                std::uint64_t   num_dists_pp = 0;
                std::tie(center_inds_pp, num_dists_pp) =
                    kmeans_plus_plus(sample, existing_centers, n_degenerate, n_candidates, &rng);
                n_dists_vec[t] += num_dists_pp;

                for (int u = 0; u < n_degenerate; ++u) {
                    const int cr = deg_idx[u];
                    const int sr = center_inds_pp(u);
                    new_centers.row(cr) = sample.row(sr);
                }
            }

            // Serial k-means on the sample
            double new_objective = 0.0;
            int iters_unused = 0;
            Eigen::VectorXi assign_unused;
            std::uint64_t num_dists_km = 0;
            std::tie(new_objective, iters_unused, assign_unused, num_dists_km) =
                kmeans(sample, new_centers, local_max_iters, local_tol);
            n_dists_vec[t] += num_dists_km;

            // Update per-thread timers/counters
            const double time_now = seconds_since_start(clock::now());
            running_time[t] = time_now;
            n_iters_vec[t] += 1;
            const int new_total_iters = total_iters.fetch_add(1, std::memory_order_relaxed) + 1;

            // Again take a snapshot of CURRENT global best
            best_obj = objectives[0];
            for (int u = 1; u < n_threads; ++u) {
                if (objectives[u] < best_obj) best_obj = objectives[u];
            }

            // Accept only if we beat the SNAPSHOT best objective (collective rule)
            if (new_objective < best_obj) {
                objectives[t]  = new_objective;
                centers_vec[t] = new_centers;
                best_times[t]  = time_now;
                best_n_iters[t]= new_total_iters;

                if (printing) {
                    std::cout << std::left << std::setw(30) << std::setprecision(6) << std::fixed << new_objective
                              << std::setw(15) << new_total_iters
                              << std::setw(15) << std::setprecision(2) << std::fixed << time_now
                              << "\n";
                }
            }

            // Stop if limits reached (redundant with loop guards—keeps latency low)
            if (max_iter > 0 && total_iters.load(std::memory_order_relaxed) >= max_iter) break;
            if (tmax > 0.0 && time_now >= tmax) break;
        }
    } // end parallel region

    // Pick globally best thread
    int best_ind = 0;
    for (int t = 1; t < n_threads; ++t) {
        if (objectives[t] < objectives[best_ind]) best_ind = t;
    }
    Eigen::MatrixXd final_centers = centers_vec[best_ind];

    // Final assignment-only pass (max_iters=0) with parallel k-means for speed
    double full_objective = 0.0;
    int iters_full = 0;
    Eigen::VectorXi assignment;
    std::uint64_t num_dists_full = 0;
    std::tie(full_objective, iters_full, assignment, num_dists_full) =
        kmeans_parallel(points, final_centers, 0, 0.0);

    // Reductions/outputs
    std::uint64_t total_dists = num_dists_full;
    for (int t = 0; t < n_threads; ++t) total_dists += n_dists_vec[t];
    int total_iters_final = total_iters.load();

    return {final_centers, full_objective, assignment,
            total_iters_final, best_n_iters[best_ind], best_times[best_ind], total_dists};
}


// Big-means with 'Hybrid Parallelism (competitive + collective)': 
// This parallelization approach involves two consecutive phases: competitive and collective.
// During the first phase, each worker tries independently to obtain its own best solution. 
// Then, during the second phase, the workers begin sharing information about the best solutions 
// with each other and try to improve them. 
// Finally, the best solution among all workers is selected as the final result.
// Additional Parameters:
// max_iter1 : Maximum number of samples to be processed for the first phase.
// max_iter2 : Maximum number of samples to be processed for the second phase.
// tmax1 : The time limit for the first phase (in seconds).
// tmax2 : The time limit for the second phase (in seconds).
// Returns: (centers, full_objective, assignment, total_iters, best_n_iter, best_time, total_dists)
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_hybrid(const Eigen::MatrixXd& points,
                 int   n_centers,
                 int   sample_size,
                 int   max_iter1,
                 int   max_iter2,
                 double tmax1,
                 double tmax2,
                 int   local_max_iters,
                 double local_tol,
                 int   n_candidates,
                 bool  printing,
                 std::mt19937_64* rng_ptr)
{
    using clock = std::chrono::steady_clock;

    const int n_points   = static_cast<int>(points.rows());
    const int n_features = static_cast<int>(points.cols());
    if (!(sample_size <= n_points) || n_points == 0 || n_features == 0 ||
        n_centers <= 0 || max_iter1 <= 0 || max_iter2 <= 0 || tmax1 <= 0.0 || tmax2 <= 0.0) {
        Eigen::MatrixXd empty_centers = Eigen::MatrixXd::Constant(
            std::max(0, n_centers), std::max(0, n_features),
            std::numeric_limits<double>::quiet_NaN());
        return {empty_centers, std::numeric_limits<double>::infinity(),
                Eigen::VectorXi(), 0, 0, 0.0, 0};
    }

    // Threads
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // Global start time
    const auto t0 = clock::now();
    auto seconds_since_start = [&](const clock::time_point& now) -> double {
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - t0).count();
    };

    // Per-thread state
    std::vector<Eigen::MatrixXd> centers_vec(n_threads, Eigen::MatrixXd::Constant(
        n_centers, n_features, std::numeric_limits<double>::quiet_NaN()));
    std::vector<double>        objectives(n_threads, std::numeric_limits<double>::infinity());
    std::vector<std::uint64_t> n_dists_vec(n_threads, 0);
    std::vector<int>           n_iters_vec(n_threads, 0);
    std::vector<double>        running_time(n_threads, 0.0);
    std::vector<double>        best_times(n_threads, 0.0);
    std::vector<int>           best_n_iters(n_threads, 0);

    // Global iteration counter (sum of all thread iterations)
    std::atomic<int> total_iters{0};

    if (printing) {
        std::cout << std::left << std::setw(30) << "sample objective"
                  << std::setw(15) << "n_iter"
                  << std::setw(15) << "cpu_time" << "\n";
    }

    // Helpers
    auto sample_indices = [](int N, int k, std::mt19937_64& rng) -> Eigen::VectorXi {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        for (int i = 0; i < k; ++i) {
            std::uniform_int_distribution<int> dist(i, N - 1);
            const int j = dist(rng);
            std::swap(idx[i], idx[j]);
        }
        Eigen::VectorXi out(k);
        for (int i = 0; i < k; ++i) out(i) = idx[i];
        return out;
    };

    auto is_row_nan = [&](const Eigen::MatrixXd& M, int r) -> bool {
        for (int c = 0; c < M.cols(); ++c) if (std::isnan(M(r, c))) return true;
        return false;
    };

    // Seed base (deterministic; optionally derived from user RNG)
    std::uint64_t base_seed = 123456789ULL;
    if (rng_ptr) base_seed ^= (*rng_ptr)();

    // --------------------------
    // Phase 1: Competitive
    // --------------------------
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
#ifdef _OPENMP
        const int t = omp_get_thread_num();
#else
        const int t = 0;
#endif
        std::mt19937_64 rng(base_seed + static_cast<std::uint64_t>(t) * 0x9E3779B97F4A7C15ULL);

        while (true) {
            // Stop conditions for phase 1
            const double now = seconds_since_start(clock::now());
            if (now >= tmax1) break;
            if (total_iters.load(std::memory_order_relaxed) >= max_iter1) break;

            const Eigen::VectorXi sample_idx = sample_indices(n_points, sample_size, rng);
            const Eigen::MatrixXd sample     = gather_rows(points, sample_idx);

            // Initialize from thread's own best
            Eigen::MatrixXd new_centers = centers_vec[t];

            // Fill degenerate rows via serial k-means++
            std::vector<int> deg_idx, good_idx;
            deg_idx.reserve(n_centers);
            good_idx.reserve(n_centers);
            for (int r = 0; r < n_centers; ++r) {
                (is_row_nan(new_centers, r) ? deg_idx : good_idx).push_back(r);
            }
            if (!deg_idx.empty()) {
                Eigen::MatrixXd existing_centers;
                if (!good_idx.empty()) {
                    Eigen::VectorXi g(static_cast<int>(good_idx.size()));
                    for (int i = 0; i < g.size(); ++i) g(i) = good_idx[i];
                    existing_centers = gather_rows(new_centers, g);
                } else {
                    existing_centers.resize(0, n_features);
                }
                Eigen::VectorXi center_inds_pp;
                std::uint64_t   num_dists_pp = 0;
                std::tie(center_inds_pp, num_dists_pp) =
                    kmeans_plus_plus(sample, existing_centers, static_cast<int>(deg_idx.size()), n_candidates, &rng);
                n_dists_vec[t] += num_dists_pp;
                for (int u = 0; u < static_cast<int>(deg_idx.size()); ++u) {
                    new_centers.row(deg_idx[u]) = sample.row(center_inds_pp(u));
                }
            }

            // Serial k-means on the sample
            double new_objective = 0.0;
            int iters_unused = 0;
            Eigen::VectorXi assign_unused;
            std::uint64_t num_dists_km = 0;
            std::tie(new_objective, iters_unused, assign_unused, num_dists_km) =
                kmeans(sample, new_centers, local_max_iters, local_tol);
            n_dists_vec[t] += num_dists_km;

            const double time_now = seconds_since_start(clock::now());
            running_time[t] = time_now;
            n_iters_vec[t] += 1;
            const int new_total_iters = total_iters.fetch_add(1, std::memory_order_relaxed) + 1;

            // For printing: compare against current global best (like Python branch)
            double best_objective_global = objectives[0];
            for (int u = 1; u < n_threads; ++u)
                if (objectives[u] < best_objective_global) best_objective_global = objectives[u];

            if (new_objective < objectives[t]) {
                objectives[t]  = new_objective;
                centers_vec[t] = new_centers;
                best_times[t]  = time_now;
                best_n_iters[t]= new_total_iters;

                if (printing && new_objective < best_objective_global) {
                    std::cout << std::left << std::setw(30) << std::setprecision(6) << std::fixed << new_objective
                              << std::setw(15) << new_total_iters
                              << std::setw(15) << std::setprecision(2) << std::fixed << time_now
                              << "\n";
                }
            }
        }
    } // end Phase 1 region

    // --------------------------
    // Phase 2: Collective
    // --------------------------
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
#ifdef _OPENMP
        const int t = omp_get_thread_num();
#else
        const int t = 0;
#endif
        std::mt19937_64 rng(base_seed + 0xA5A5A5A5FFFFULL + static_cast<std::uint64_t>(t) * 0x9E3779B97F4A7C15ULL);

        while (true) {
            // Stop conditions for full hybrid (phase1+phase2)
            const double now = seconds_since_start(clock::now());
            if (now >= (tmax1 + tmax2)) break;
            if (total_iters.load(std::memory_order_relaxed) >= (max_iter1 + max_iter2)) break;

            const Eigen::VectorXi sample_idx = sample_indices(n_points, sample_size, rng);
            const Eigen::MatrixXd sample     = gather_rows(points, sample_idx);

            // Snapshot: current global best index & objective
            int best_idx = 0;
            double best_obj = objectives[0];
            for (int u = 1; u < n_threads; ++u) {
                if (objectives[u] < best_obj) { best_obj = objectives[u]; best_idx = u; }
            }

            // Initialize from global best centers
            Eigen::MatrixXd new_centers = centers_vec[best_idx];

            // Fill degenerate rows via serial k-means++
            std::vector<int> deg_idx, good_idx;
            deg_idx.reserve(n_centers);
            good_idx.reserve(n_centers);
            for (int r = 0; r < n_centers; ++r) {
                (is_row_nan(new_centers, r) ? deg_idx : good_idx).push_back(r);
            }
            if (!deg_idx.empty()) {
                Eigen::MatrixXd existing_centers;
                if (!good_idx.empty()) {
                    Eigen::VectorXi g(static_cast<int>(good_idx.size()));
                    for (int i = 0; i < g.size(); ++i) g(i) = good_idx[i];
                    existing_centers = gather_rows(new_centers, g);
                } else {
                    existing_centers.resize(0, n_features);
                }
                Eigen::VectorXi center_inds_pp;
                std::uint64_t   num_dists_pp = 0;
                std::tie(center_inds_pp, num_dists_pp) =
                    kmeans_plus_plus(sample, existing_centers, static_cast<int>(deg_idx.size()), n_candidates, &rng);
                n_dists_vec[t] += num_dists_pp;
                for (int u = 0; u < static_cast<int>(deg_idx.size()); ++u) {
                    new_centers.row(deg_idx[u]) = sample.row(center_inds_pp(u));
                }
            }

            // Serial k-means on the sample
            double new_objective = 0.0;
            int iters_unused = 0;
            Eigen::VectorXi assign_unused;
            std::uint64_t num_dists_km = 0;
            std::tie(new_objective, iters_unused, assign_unused, num_dists_km) =
                kmeans(sample, new_centers, local_max_iters, local_tol);
            n_dists_vec[t] += num_dists_km;

            const double time_now = seconds_since_start(clock::now());
            running_time[t] = time_now;
            n_iters_vec[t] += 1;
            const int new_total_iters = total_iters.fetch_add(1, std::memory_order_relaxed) + 1;

            // Accept only if beating the snapshot global best (collective rule)
            if (new_objective < best_obj) {
                objectives[t]  = new_objective;
                centers_vec[t] = new_centers;
                best_times[t]  = time_now;
                best_n_iters[t]= new_total_iters;

                if (printing) {
                    std::cout << std::left << std::setw(30) << std::setprecision(6) << std::fixed << new_objective
                              << std::setw(15) << new_total_iters
                              << std::setw(15) << std::setprecision(2) << std::fixed << time_now
                              << "\n";
                }
            }

            if (now >= (tmax1 + tmax2)) break;
            if (total_iters.load(std::memory_order_relaxed) >= (max_iter1 + max_iter2)) break;
        }
    } // end Phase 2 region

    // Select global best
    int best_ind = 0;
    for (int t = 1; t < n_threads; ++t)
        if (objectives[t] < objectives[best_ind]) best_ind = t;

    Eigen::MatrixXd final_centers = centers_vec[best_ind];

    // Final assignment-only pass
    double full_objective = 0.0;
    int iters_full = 0;
    Eigen::VectorXi assignment;
    std::uint64_t num_dists_full = 0;
    std::tie(full_objective, iters_full, assignment, num_dists_full) =
        kmeans_parallel(points, final_centers, 0, 0.0);

    // Reductions
    std::uint64_t total_dists = num_dists_full;
    for (int t = 0; t < n_threads; ++t) total_dists += n_dists_vec[t];
    int total_iters_final = total_iters.load();

    return {final_centers, full_objective, assignment,
            total_iters_final, best_n_iters[best_ind], best_times[best_ind], total_dists};
}


    
} // namespace bigmeans
