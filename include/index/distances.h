#ifndef MICROVECDB_DISTANCES_H
#define MICROVECDB_DISTANCES_H

#include "constants.h"

#include <iostream>
#include <cmath>
#include <cstddef>

#ifdef __AVX2__
#include <immintrin.h> // AVX2 intrinsics
#elif defined(__AVX__)
#include <immintrin.h> // AVX intrinsics
#elif defined(__ARM_NEON)
#include <arm_neon.h> // NEON intrinsics
#endif

#include <cblas.h> // BLAS

namespace mvdb {

    float l2_distance_optimised_single(const float* vec1, const float* vec2, const size_t& d);

    /**
     * Computes the L2 (Euclidean) distance between vectors in two sets, optimized with SIMD and OpenMP.
     *
     * @param vec Array of vectors.
     * @param n Number of vectors in vec.
     * @param d Dimensionality of each vector.
     * @param vec_comp Array of vectors to compare against.
     * @param vec_comp_n Number of vectors in vec_comp.
     * @return Dynamically allocated array of distances, size n * vec_comp_n.
    */
    float *l2_distance_optimised(const float *vec, size_t n, size_t d, const float *vec_comp, size_t vec_comp_n);

    /**
     * Computes the squared L2 distance between two vectors using BLAS.
     * This helper function calculates ||vec1 - vec2||^2.
     */
    float l2_distance_naive_single(const float *vec1, const float *vec2, const size_t& d);

    /**
     * Computes the L2 distance between sets of vectors using no extra optimisations.
     * @param vec Array of the first set of vectors.
     * @param n Number of vectors in the first set.
     * @param d Dimensionality of vectors.
     * @param vec_comp Array of the second set of vectors to compare against.
     * @param vec_comp_n Number of vectors in the second set.
     * @return Dynamically allocated array of distances of size n * vec_comp_n.
     */
    float *l2_distance_naive(const float *vec, size_t n, size_t d, const float *vec_comp, size_t vec_comp_n);

    /**
     * Computes the L2 distances between two sets of vectors using BLAS for optimization.
     *
     * @param vec Array of the first set of vectors, each of dimension d.
     * @param n Number of vectors in the first set.
     * @param d Dimensionality of the vectors.
     * @param vec_comp Array of the second set of vectors to compare against, each of dimension d.
     * @param vec_comp_n Number of vectors in the second set.
     * @return Dynamically allocated array of distances, size n * vec_comp_n.
    */
    float *l2_distance_blas(const float *vec, size_t n, size_t d, const float *vec_comp, size_t vec_comp_n);

} // namespace mvdb

#endif //MICROVECDB_DISTANCES_H
