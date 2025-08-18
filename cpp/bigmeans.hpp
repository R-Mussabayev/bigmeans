#pragma once
#ifndef BIGMEANS_HPP
#define BIGMEANS_HPP

#include <Eigen/Dense>
#include <tuple>
#include <utility>
#include <random>
#include <cstdint>

namespace bigmeans {

// ---------- Distances ----------

/**
 * @brief Squared Euclidean distance matrix (serial).
 *        Returns an X.rows() x Y.rows() matrix with entries ||X_i - Y_j||^2.
 */
Eigen::MatrixXd distance_mat(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);

/**
 * @brief Squared Euclidean distance matrix (parallel over Y rows when OpenMP is enabled).
 *        Returns an X.rows() x Y.rows() matrix with entries ||X_i - Y_j||^2.
 */
Eigen::MatrixXd distance_mat_parallel(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);


// ---------- K-means++ initializers ----------

/**
 * @brief Serial K-means++ initializer (no OpenMP inside).
 * @param points         (N x d) dataset
 * @param centers        (c x d) existing centers (c can be 0)
 * @param n_new_centers  number of new centers to select (>= 2)
 * @param n_candidates   number of candidates per step (>= 1)
 * @param rng_ptr        optional RNG to use; if null, a local deterministic RNG is used
 * @return (indices_of_chosen_points, proxy_distance_count)
 */
std::pair<Eigen::VectorXi, std::uint64_t>
kmeans_plus_plus(const Eigen::MatrixXd& points,
                 const Eigen::MatrixXd& centers,
                 int n_new_centers = 2,
                 int n_candidates  = 6,
                 std::mt19937_64* rng_ptr = nullptr);

/**
 * @brief Parallel K-means++ initializer (uses distance_mat_parallel internally).
 * @return (indices_of_chosen_points, proxy_distance_count)
 */
std::pair<Eigen::VectorXi, std::uint64_t>
kmeans_plus_plus_parallel(const Eigen::MatrixXd& points,
                          const Eigen::MatrixXd& centers,
                          int n_new_centers = 2,
                          int n_candidates  = 6,
                          std::mt19937_64* rng_ptr = nullptr);


// ---------- K-means solvers ----------

/**
 * @brief Serial K-means (no OpenMP inside).
 * @param points   (N x d) dataset
 * @param centers  (K x d) centers (modified in-place)
 * @param max_iters  maximum iterations; if 0, run assignment-only
 * @param tol        relative improvement threshold to stop
 * @return (objective, n_iters, assignment, proxy_op_count)
 */
std::tuple<double, int, Eigen::VectorXi, std::uint64_t>
kmeans(const Eigen::MatrixXd& points,
       Eigen::MatrixXd&       centers,
       int                    max_iters = -1,
       double                 tol       = 0.0);

/**
 * @brief Parallel K-means (parallelizes the assignment step with OpenMP when enabled).
 * @return (objective, n_iters, assignment, proxy_op_count)
 */
std::tuple<double, int, Eigen::VectorXi, std::uint64_t>
kmeans_parallel(const Eigen::MatrixXd& points,
                Eigen::MatrixXd&       centers,
                int                    max_iters = -1,
                double                 tol       = 0.0);


// ---------- Big-means variants ----------

/**
 * @brief Fully sequential Big-means (no OpenMP inside inner routines).
 * @return (centers, full_objective, assignment, n_iter, best_n_iter, best_time, total_dists)
 */
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_sequential(const Eigen::MatrixXd& points,
                     int   n_centers       = 3,
                     int   sample_size     = 100,
                     int   max_iter        = 10000,
                     double tmax           = 10.0,
                     int   local_max_iters = 300,
                     double local_tol      = 1e-4,
                     int   n_candidates    = 3,
                     bool  printing        = false,
                     std::mt19937_64* rng_ptr = nullptr);

/**
 * @brief Big-means with Inner Parallelism (K-means / K-means++ themselves are parallel).
 * @return (centers, full_objective, assignment, n_iter, best_n_iter, best_time, total_dists)
 */
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_inner(const Eigen::MatrixXd& points,
                int   n_centers       = 3,
                int   sample_size     = 100,
                int   max_iter        = 10000,
                double tmax           = 10.0,
                int   local_max_iters = 300,
                double local_tol      = 1e-4,
                int   n_candidates    = 3,
                bool  printing        = false,
                std::mt19937_64* rng_ptr = nullptr);

/**
 * @brief Big-means with Competitive Parallelism (workers improve independently).
 * @return (centers, full_objective, assignment, total_iters, best_n_iter, best_time, total_dists)
 */
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_competitive(const Eigen::MatrixXd& points,
                      int   n_centers       = 3,
                      int   sample_size     = 100,
                      int   max_iter        = 10000,
                      double tmax           = 10.0,
                      int   local_max_iters = 300,
                      double local_tol      = 1e-4,
                      int   n_candidates    = 3,
                      bool  printing        = false,
                      std::mt19937_64* rng_ptr = nullptr);

/**
 * @brief Big-means with Collective Parallelism (workers initialize from global-best).
 * @return (centers, full_objective, assignment, total_iters, best_n_iter, best_time, total_dists)
 */
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_collective(const Eigen::MatrixXd& points,
                     int   n_centers       = 3,
                     int   sample_size     = 100,
                     int   max_iter        = 10000,
                     double tmax           = 10.0,
                     int   local_max_iters = 300,
                     double local_tol      = 1e-4,
                     int   n_candidates    = 3,
                     bool  printing        = false,
                     std::mt19937_64* rng_ptr = nullptr);

/**
 * @brief Big-means with Hybrid Parallelism (competitive phase then collective phase).
 * @return (centers, full_objective, assignment, total_iters, best_n_iter, best_time, total_dists)
 */
std::tuple<Eigen::MatrixXd, double, Eigen::VectorXi, int, int, double, std::uint64_t>
big_means_hybrid(const Eigen::MatrixXd& points,
                 int   n_centers        = 3,
                 int   sample_size      = 100,
                 int   max_iter1        = 10000,
                 int   max_iter2        = 10000,
                 double tmax1           = 10.0,
                 double tmax2           = 10.0,
                 int   local_max_iters  = 300,
                 double local_tol       = 1e-4,
                 int   n_candidates     = 3,
                 bool  printing         = false,
                 std::mt19937_64* rng_ptr = nullptr);

} // namespace bigmeans

#endif // BIGMEANS_HPP
