#include "distances.h"
#include <cblas.h>

namespace mvdb {

    float* l2_distance_intrinsics(const float* vec, size_t n, size_t d, const float* vec_comp, size_t vec_comp_n) {
        auto* distances = new float[n * vec_comp_n];

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < vec_comp_n; ++j) {
                float sum = 0.0f;

                #if defined(__AVX2__) || defined(__AVX__)
                std::cout << "AVX or AVX2" << std::endl;
                    for (size_t k = 0; k <= d - 8; k += 8) {
                        __m256 vec_chunk = _mm256_loadu_ps(&vec[i * d + k]);
                        __m256 vec_comp_chunk = _mm256_loadu_ps(&vec_comp[j * d + k]);
                        __m256 diff = _mm256_sub_ps(vec_chunk, vec_comp_chunk);
                        __m256 sq_diff = _mm256_mul_ps(diff, diff);
                        __m256 sum_vec = _mm256_hadd_ps(sq_diff, sq_diff); // Horizontal add
                        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec); // Second hadd for reduction
                        float temp[8];
                        _mm256_storeu_ps(temp, sum_vec);
                        sum += temp[0] + temp[4];
                    }
                #elif defined(__ARM_NEON)
                std::cout << "ARM_NEON" << std::endl;
                    size_t k = 0;
                    for (; k <= d - 4; k += 4) {
                        float32x4_t vec_chunk = vld1q_f32(&vec[i * d + k]);
                        float32x4_t vec_comp_chunk = vld1q_f32(&vec_comp[j * d + k]);
                        float32x4_t diff = vsubq_f32(vec_chunk, vec_comp_chunk);
                        float32x4_t sq_diff = vmulq_f32(diff, diff);
                        float32x2_t sum_vec = vpadd_f32(vget_low_f32(sq_diff), vget_high_f32(sq_diff));
                        sum += vget_lane_f32(sum_vec, 0) + vget_lane_f32(sum_vec, 1);
                    }
                #endif
                // Handle remaining elements
                for (size_t k = k; k < d; ++k) {
                    float diff = vec[i * d + k] - vec_comp[j * d + k];
                    sum += diff * diff;
                }
                distances[i * vec_comp_n + j] = std::sqrt(sum);
            }
        }
        return distances;
    }

    float squared_l2_distance_naive(const float *vec1, const float *vec2, const idx_t& d) {
        float result = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float diff = vec1[i] - vec2[i];
            result += diff * diff;
        }
        return result;
    }

    float* l2_distance_naive(const float* vec, size_t n, size_t d, const float* vec_comp, size_t vec_comp_n) {
        auto* distances = new float[n * vec_comp_n];

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < vec_comp_n; ++j) {
                // Compute the squared L2 distance using the helper function
                float sq_distance = squared_l2_distance_naive(vec + i * d, vec_comp + j * d, d);
                // Store the square root of the squared L2 distance
                distances[i * vec_comp_n + j] = std::sqrt(sq_distance);
            }
        }

        return distances;
    }

    float* l2_distance_blas(const float* vec, size_t n, size_t d, const float* vec_comp, size_t vec_comp_n) {
        auto* distances = new float[n * vec_comp_n];

        // Parallelize the outer loop using OpenMP
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < vec_comp_n; ++j) {
                // Initialize sum for the squared differences
                float sum = 0.0f;

                // Compute the dot product of vec[i] with itself
                float ai_dot = cblas_sdot(d, &vec[i * d], 1, &vec[i * d], 1);
                // Compute the dot product of vec_comp[j] with itself
                float bj_dot = cblas_sdot(d, &vec_comp[j * d], 1, &vec_comp[j * d], 1);
                // Compute the dot product of vec[i] and vec_comp[j]
                float ab_dot = cblas_sdot(d, &vec[i * d], 1, &vec_comp[j * d], 1);

                // Calculate the squared L2 distance using the dot products
                sum = ai_dot - 2 * ab_dot + bj_dot;

                // Store the square root of the sum, which is the L2 distance
                distances[i * vec_comp_n + j] = std::sqrt(sum);
            }
        }

        return distances;
    }

} // namespace mvdb