#include "bigmeans.hpp"
#include <omp.h>
#include <Eigen/Dense>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <chrono> // <-- Added for timing

int main() {
    // Load dataset: first 3 tab-separated columns
    std::ifstream fin("skin_segmentation.data");
    if (!fin) {
        std::cerr << "Error: cannot open data file\n";
        return 1; 
    }

    std::vector<std::array<double,3>> rows;
    rows.reserve(250000);
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::array<double,3> vals{};
        std::size_t start=0, pos=0; int col=0;
        for (; col<3 && (pos=line.find('\t', start))!=std::string::npos; ++col) {
            vals[col] = std::stod(line.substr(start, pos-start));
            start = pos+1;
        }
        if (col<3) { vals[col++] = std::stod(line.substr(start)); }
        if (col==3) rows.push_back(vals);
    }
    Eigen::MatrixXd points((int)rows.size(), 3);
    for (int i=0;i<(int)rows.size();++i) {
        points(i,0)=rows[i][0]; points(i,1)=rows[i][1]; points(i,2)=rows[i][2];
    }

    // Print OpenMP threads info
    std::cout << "OpenMP reports max threads: " << omp_get_max_threads() << "\n\n";

    //Big-Means Parameters
    const int    sample_size     = 160000;  // The number of data points to be randomly selected from the input dataset at each iteration of the Big-means.
    const int    n_centers       = 25; // The desired number of clusters
    const int    max_iter        = 2000; // Maximum number of samples to be processed
    const double tmax            = 1000.0; // The time limit for the search process (in seconds); a zero or negative value means no limit.
    const int    local_max_iters = 300; // The maximum number of K-means iterations before declaring convergence and stopping the clustering process for each sample.
    const double local_tol       = 1e-4; // The threshold below which the relative change in the objective function between two iterations must fall to declare convergence of K-means.
    const int    n_candidates    = 3; // The number of candidate centers to choose from at each stage of the K-means++ initialization algorithm

    
    // Best Known Solution (for comparison)
    const double f_best          = 102280000.0;



    // Ask the user which algorithm to run
    std::cout << "Choose the variant of Big-means algorithm to run:\n"
                 "  1) Sequential\n"
                 "  2) Inner Parallelism\n"
                 "  3) Competitive Parallelism\n"
                 "  4) Collective Parallelism\n"
                 "  5) Hybrid Parallelism\n"
                 "Enter a number (1-5): ";
    int choice = 0;
    std::cin >> choice;
    std::cout << "\n";

    // Prepare shared outputs & timing outside the switch
    using ResultT = std::tuple<Eigen::MatrixXd,double,Eigen::VectorXi,int,int,double,std::uint64_t>;
    ResultT result;
    std::chrono::high_resolution_clock::time_point t0, t1;
    std::string label;
    
    // Run selected algorithm (Sequential is default on invalid input)
    switch (choice) {
        case 1:
            label = "SEQUENTIAL BIG-MEANS";
            t0 = std::chrono::high_resolution_clock::now();
            result = bigmeans::big_means_sequential(points, n_centers, sample_size, max_iter,
                                                    tmax, local_max_iters, local_tol,
                                                    n_candidates, /*printing=*/true);
            t1 = std::chrono::high_resolution_clock::now();
            break;
    
        case 2:
            label = "BIG-MEANS WITH 'INNER PARALLELISM'";
            t0 = std::chrono::high_resolution_clock::now();
            result = bigmeans::big_means_inner(points, n_centers, sample_size, max_iter,
                                               tmax, local_max_iters, local_tol,
                                               n_candidates, /*printing=*/true);
            t1 = std::chrono::high_resolution_clock::now();
            break;
    
        case 3:
            label = "BIG-MEANS WITH 'COMPETITIVE PARALLELISM'";
            t0 = std::chrono::high_resolution_clock::now();
            result = bigmeans::big_means_competitive(points, n_centers, sample_size, max_iter,
                                                     tmax, local_max_iters, local_tol,
                                                     n_candidates, /*printing=*/true);
            t1 = std::chrono::high_resolution_clock::now();
            break;
    
        case 4:
            label = "BIG-MEANS WITH 'COLLECTIVE PARALLELISM'";
            t0 = std::chrono::high_resolution_clock::now();
            result = bigmeans::big_means_collective(points, n_centers, sample_size, max_iter,
                                                    tmax, local_max_iters, local_tol,
                                                    n_candidates, /*printing=*/true);
            t1 = std::chrono::high_resolution_clock::now();
            break;
    
        case 5:
            label = "BIG-MEANS WITH 'HYBRID PARALLELISM'";
            t0 = std::chrono::high_resolution_clock::now();
            result = bigmeans::big_means_hybrid(points,
                                                n_centers,
                                                sample_size,
                                                /*max_iter1=*/max_iter,
                                                /*max_iter2=*/max_iter,
                                                /*tmax1=*/tmax,
                                                /*tmax2=*/tmax,
                                                local_max_iters,
                                                local_tol,
                                                n_candidates,
                                                /*printing=*/true);
            t1 = std::chrono::high_resolution_clock::now();
            break;
    
        default:
            std::cerr << "Invalid choice. Running Sequential by default.\n";
            label = "SEQUENTIAL BIG-MEANS (default)";
            t0 = std::chrono::high_resolution_clock::now();
            result = bigmeans::big_means_sequential(points, n_centers, sample_size, max_iter,
                                                    tmax, local_max_iters, local_tol,
                                                    n_candidates, /*printing=*/true);
            t1 = std::chrono::high_resolution_clock::now();
            break;
    }
    
    // Unpack once, after the switch
    auto [centers, objective, assignment, n_iter, best_n_iter, best_time, n_dists] = std::move(result);
    

    

    // Report

    std::chrono::duration<double> elapsed = t1 - t0;

    std::cout << "\n#Iterations: " << n_iter << "\n";
    std::cout << "#Distances: "  << n_dists << "\n";
    std::cout << "Full Objective: " << std::setprecision(10) << objective << "\n";

    // Objective gap
    const double gap = ((objective - f_best)/objective)*100.0;
    std::cout << "Objective Gap: " << std::fixed << std::setprecision(2) << gap << " %\n";
    std::cout << "Best at iteration " << best_n_iter
              << " (elapsed ~" << std::fixed << std::setprecision(2) << best_time << " s)\n";

    // Print total runtime
    std::cout << "Total runtime: " 
              << std::fixed << std::setprecision(2) << elapsed.count() << " s\n";
    
    (void)centers; (void)assignment; // if unused further
    return 0;
}




