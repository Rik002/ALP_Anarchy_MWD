#include "Simulation.h"
#include "MatrixUtils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <Eigen/Dense>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// INI parser remains the same...
std::map<std::string, std::string> parseINI(const std::string& filename) {
    std::map<std::string, std::string> config;
    std::ifstream file(filename);
    std::string line, section;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line.find("---") != std::string::npos) continue;

        if (line[0] == '[') {
            section = line.substr(1, line.size() - 2);
            continue;
        }

        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));

            size_t colon_pos = value.find(':');
            if (colon_pos != std::string::npos) {
                value = value.substr(0, colon_pos);
                value.erase(value.find_last_not_of(" \t") + 1);
            }

            config[section + "." + key] = value;
        }
    }
    return config;
}

Simulation::Simulation(const std::string& config_file) : rng(std::random_device()()) {
    auto config = parseINI(config_file);
    try {
        b_N1 = std::stod(config["Simulation.b_N1"]);
        N_min = std::stoi(config["Simulation.N_min"]);
        N_max = std::stoi(config["Simulation.N_max"]);
        N_real = std::stoi(config["Simulation.N_real"]);
        L = std::stod(config["Simulation.L"]);
        E = std::stod(config["Simulation.E"]);
        m_min = std::stod(config["Simulation.m_min"]);
        m_max = std::stod(config["Simulation.m_max"]);
        precision = std::stoi(config["Simulation.precision"]);
        data_dir = config["Directories.data_dir"];
    } catch (const std::exception& e) {
        throw std::runtime_error("Error parsing config file: " + std::string(e.what()));
    }
}

Simulation::~Simulation() {
    cleanupThreadRngs();
}

void Simulation::initializeThreadRngs(unsigned int master_seed) {
    int num_threads = omp_get_max_threads();
    thread_rngs.resize(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        thread_rngs[i] = MatrixUtils::initializeGSLRng(master_seed + i * 10000);
    }
}

void Simulation::cleanupThreadRngs() {
    for (auto* rng : thread_rngs) {
        if (rng) {
            MatrixUtils::destroyGSLRng(rng);
        }
    }
    thread_rngs.clear();
}

void Simulation::runEigenOptimized() {
    // Get available threads
    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads with Eigen + GSL optimization" << std::endl;
    std::cout << "Starting ALP Anarchy simulation with b_N1 = " << std::scientific << b_N1 << std::endl;
    std::cout << "N range: " << N_min << " to " << N_max << std::endl;
    std::cout << "Realizations per N: " << N_real << std::endl;

    // Master random seed for reproducibility
    std::random_device rd;
    unsigned int master_seed = rd();
    std::cout << "Master seed: " << master_seed << std::endl;

    // Initialize GSL random number generators for all threads
    initializeThreadRngs(master_seed);

    for (int N = N_min; N <= N_max; ++N) {
        std::vector<double> bounds(N_real);
        std::cout << "Processing N = " << N << " (Eigen+GSL parallel)..." << std::flush;

        auto start_time = std::chrono::high_resolution_clock::now();

        // PARALLEL REGION with Eigen and GSL
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            gsl_rng* local_rng = thread_rngs[thread_id];

            // Set thread-specific seed for reproducibility
            gsl_rng_set(local_rng, master_seed + N * 1000 + thread_id);

            #pragma omp for schedule(static)
            for (int i = 0; i < N_real; ++i) {
                // Generate two independent SO(N) matrices using Eigen
               // Eigen::MatrixXd U_gamma = MatrixUtils::generateSONMatrixEigen(N, local_rng);
               // Eigen::MatrixXd U_e = MatrixUtils::generateSONMatrixEigen(N, local_rng);
               // Eigen::MatrixXd U_gamma = MatrixUtils::generateSONMatrix(N, local_rng);
               // Eigen::MatrixXd U_e = MatrixUtils::generateSONMatrix(N, local_rng);

               // NEW (use GSL version and convert to Eigen):
               auto U_gamma_vec = MatrixUtils::generateSONMatrixGSL(N, local_rng);
               auto U_e_vec = MatrixUtils::generateSONMatrixGSL(N, local_rng);

               // Convert std::vector<std::vector<double>> to Eigen::MatrixXd
               Eigen::MatrixXd U_gamma(N, N);
               Eigen::MatrixXd U_e(N, N);

               for (int i = 0; i < N; ++i) {
                   for (int j = 0; j < N; ++j) {
                       U_gamma(i, j) = U_gamma_vec[i][j];
                       U_e(i, j) = U_e_vec[i][j];
                   }
               }

                // Extract first rows as Eigen vectors
                Eigen::VectorXd g_gamma_mass = U_gamma.row(0);
                Eigen::VectorXd g_e_mass = U_e.row(0);

                // Dot product using Eigen (vectorized and optimized)
                double dot_product = MatrixUtils::dotProductEigen(g_gamma_mass, g_e_mass);

                // Compute total effective couplings
                double g_gamma_total = 0.0, g_e_total = 0.0, g_gamma_sq=0.0, g_e_sq=0.0;
                for (int j = 0; j < N; ++j) {
                    g_gamma_total += g_gamma_mass[j] * g_gamma_mass[j];
                    g_e_total += g_e_mass[j] * g_e_mass[j];
                }
                g_gamma_sq = std::sqrt(g_gamma_total);
                g_e_sq = std::sqrt(g_e_total);

                // Conversion probability: P = |dot_product|²/(|g_gamma|²|g_e|²)
                double P = (dot_product * dot_product);///(g_gamma_total*g_e_total);

                // Thread-safe debug output
                if (N == N_min && i < 3 && thread_id == 0) {
                    #pragma omp critical
                    {
                        std::cout << "\nEigen+GSL Debug N=" << N << ", i=" << i << ":" << std::endl;
                        std::cout << "  First row U_gamma: ";
                        for (int j = 0; j < std::min(3, N); ++j) {
                            std::cout << g_gamma_mass(j) << " ";
                        }
                        std::cout << std::endl;
                        std::cout << "  dot_product = " << dot_product << std::endl;
                        std::cout << "  P = " << P << std::endl;
                        std::cout << "  Expected <P> ≈ " << (1.0/N) << std::endl;
                        std::cout << "  Matrix condition: " << U_gamma.norm() << std::endl;
                    }
                }

                // Avoid division by zero
                if (P < 1e-20L) P = 1e-20L;

                // MWD bound
                bounds[i] = b_N1 / P;

                // Sanity check
                if (!std::isfinite(bounds[i]) || bounds[i] <= 0) {
                    #pragma omp critical
                    {
                        std::cerr << "ERROR: Invalid bound at N=" << N << ", i=" << i
                                  << ", thread=" << thread_id << std::endl;
                    }
                }
            }
        } // End parallel region

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << " done. (" << duration.count() << " ms)" << std::endl;

        performStatisticalTests(N, bounds);
        saveBounds(N, bounds);
    }

    std::cout << "Eigen+GSL simulation completed successfully!" << std::endl;
}

// Legacy run method for compatibility
void Simulation::run() {
    runEigenOptimized();  // Use the optimized version by default
}

// Statistical tests and file I/O remain the same...
void Simulation::performStatisticalTests(int N, const std::vector<double>& bounds) {
    double mean = 0.0, variance = 0.0;
    for (double b : bounds) mean += b;
    mean /= bounds.size();
    for (double b : bounds) variance += std::pow(b - mean, 2);
    variance /= (bounds.size() - 1);

    std::string logfile = data_dir + "stats_N" + std::to_string(N) + ".txt";
    std::ofstream log(logfile);
    log << std::fixed << std::setprecision(precision);
    log << "# Statistical Tests Results for N = " << N << std::endl;
    log << "# Generated by ALP Anarchy Simulation (Eigen+GSL)" << std::endl;
    log << "#" << std::endl;
    log << "N_ALPs: " << N << std::endl;
    log << "N_realizations: " << bounds.size() << std::endl;
    log << "Mean_bound: " << std::scientific << mean << std::endl;
    log << "Std_deviation: " << std::scientific << std::sqrt(variance) << std::endl;
    log << "Min_bound: " << std::scientific << *std::min_element(bounds.begin(), bounds.end()) << std::endl;
    log << "Max_bound: " << std::scientific << *std::max_element(bounds.begin(), bounds.end()) << std::endl;
    log.close();
}

void Simulation::saveBounds(int N, const std::vector<double>& bounds) {
    std::string filename = data_dir + "bounds_N" + std::to_string(N) + ".txt";
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file " << filename << std::endl;
        return;
    }

    file << "# Multi-ALP Bounds Data File" << std::endl;
    file << "# Generated by ALP Anarchy Simulation (Eigen+GSL+OpenMP)" << std::endl;
    file << "# N_ALPs: " << N << std::endl;
    file << "# N_realizations: " << bounds.size() << std::endl;
    file << "# b_N1: " << std::scientific << std::setprecision(6) << b_N1 << std::endl;
    file << "# Libraries: Eigen " << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION
         << ", GSL, OpenMP" << std::endl;
    file << "# Columns: Realization_Index, Bound_Value" << std::endl;
    file << "#" << std::endl;

    for (size_t i = 0; i < bounds.size(); ++i) {
        file << std::setw(6) << i+1 << "  " << std::scientific
             << std::setprecision(precision) << bounds[i] << std::endl;
    }

    file.flush();
    file.close();
    std::cout << "✓ Saved " << bounds.size() << " bounds for N=" << N << std::endl;
}
