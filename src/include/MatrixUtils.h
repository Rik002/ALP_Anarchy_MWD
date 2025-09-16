#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>
#include <random>

// DISABLE MKL SUPPORT TO AVOID CONFLICTS
//#define EIGEN_DONT_USE_MKL
//#define EIGEN_DONT_USE_BLAS
//#define EIGEN_DONT_USE_LAPACKE

#include <Eigen/Dense>
#include <Eigen/QR>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

class MatrixUtils {
public:
    // Generate an SO(N) matrix using the Haar measure (Eigen-optimized)
    static Eigen::MatrixXd generateSONMatrixEigen(int N, gsl_rng* rng);

    // Legacy version with std::mt19937
    static std::vector<std::vector<double>> generateSONMatrix(int N, std::mt19937& rng);

    // NEW: Legacy version with GSL RNG
    static std::vector<std::vector<double>> generateSONMatrixGSL(int N, gsl_rng* rng);

    // Eigen-optimized vector operations
    static double dotProductEigen(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2);
    static Eigen::VectorXd normalizeVectorEigen(const Eigen::VectorXd& v);

    // Legacy versions
    static double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2);
    static std::vector<double> normalizeVector(const std::vector<double>& v);

    // GSL random number utilities
    static gsl_rng* initializeGSLRng(unsigned long seed);
    static void destroyGSLRng(gsl_rng* rng);

    // MISSING FUNCTION: Add determinant computation
    static double computeDeterminant(const std::vector<std::vector<double>>& matrix);
};

#endif // MATRIX_UTILS_H
