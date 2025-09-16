#ifndef SIMULATION_H
#define SIMULATION_H

#include <string>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <gsl/gsl_rng.h>

class Simulation {
private:
    double b_N1;
    int N_min, N_max, N_real;
    double L, E, m_min, m_max;
    int precision;
    std::string data_dir;
    std::mt19937 rng;

    // GSL random number generators for parallel threads
    std::vector<gsl_rng*> thread_rngs;

public:
    Simulation(const std::string& config_file);
    ~Simulation();  // Destructor to clean up GSL resources

    void run();
    void runEigenOptimized();  // New Eigen-optimized version

    void performStatisticalTests(int N, const std::vector<double>& bounds);
    void saveBounds(int N, const std::vector<double>& bounds);

private:
    void initializeThreadRngs(unsigned int master_seed);
    void cleanupThreadRngs();
};

#endif // SIMULATION_H
