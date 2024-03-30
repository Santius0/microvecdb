#include "distances.h"

#if defined(__CUDA__)
    #include <cuda_runtime.h>

    __global__ void squareDifferences(float *result, const float *vec1, const float *vec2, int d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < d) {
        float diff = vec1[idx] - vec2[idx];
        result[idx] = diff * diff;
    }
}
#endif

namespace mvdb {

    float l2_distance_naive_single(const float *vec1, const float *vec2, const size_t& d) {
        float sum = 0.0f, diff;
        for (size_t i = 0; i < d; ++i) {
            diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    float* l2_distance_naive(const float* vec, size_t n, size_t d, const float* vec_comp, size_t vec_comp_n) {
        auto* distances = new float[n * vec_comp_n];
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < vec_comp_n; ++j) {
                distances[i * vec_comp_n + j] = l2_distance_naive_single(vec + i * d, vec_comp + j * d, d);
            }
        }
        return distances;
    }


    float l2_distance_blas_single(const float *vec1, const float *vec2, const size_t& d) {
        // Compute the dot product of vec[i] with itself
        float ai_dot = cblas_sdot(static_cast<int>(d), &vec1[0], 1, &vec1[0], 1);
        // Compute the dot product of vec_comp[j] with itself
        float bj_dot = cblas_sdot(static_cast<int>(d), &vec2[0], 1, &vec2[0], 1);
        // Compute the dot product of vec[i] and vec_comp[j]
        float ab_dot = cblas_sdot(static_cast<int>(d), &vec1[0], 1, &vec2[0], 1);
        // Calculate the squared L2 distance using the dot products
        float sum = ai_dot - 2 * ab_dot + bj_dot;
        return std::sqrt(sum);
    }

    float* l2_distance_blas(const float* vec, size_t n, size_t d, const float* vec_comp, size_t vec_comp_n) {
        auto* distances = new float[n * vec_comp_n];
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < vec_comp_n; ++j) {
                distances[i * vec_comp_n + j] = l2_distance_blas_single(vec + i * d, vec_comp + j * d, d);
            }
        }
        return distances;
    }


    float l2_distance_optimised_single(const float* vec1, const float* vec2, const size_t& d) {
        float sum = 0.0f;

        #if defined(__CUDA__)
            float *dev_vec1, *dev_vec2, *dev_result;
            float *result = (float*)malloc(d * sizeof(float));

            cudaMalloc(&dev_vec1, d * sizeof(float));
            cudaMalloc(&dev_vec2, d * sizeof(float));
            cudaMalloc(&dev_result, d * sizeof(float));

            cudaMemcpy(dev_vec1, vec1, d * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_vec2, vec2, d * sizeof(float), cudaMemcpyHostToDevice);

            int threadsPerBlock = 256;
            int blocks = (d + threadsPerBlock - 1) / threadsPerBlock;
            squareDifferences<<<blocks, threadsPerBlock>>>(dev_result, dev_vec1, dev_vec2, d);

            cudaMemcpy(result, dev_result, d * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < d; i++) sum += result[i];

            cudaFree(dev_vec1);
            cudaFree(dev_vec2);
            cudaFree(dev_result);
            free(result);
        #else
            size_t k;
            #if defined(__AVX2__) || defined(__AVX__)
                for (k = 0; k <= d - 8; k += 8) {
                    __m256 vec1_chunk = _mm256_loadu_ps(&vec1[k]);
                    __m256 vec2_chunk = _mm256_loadu_ps(&vec2[k]);
                    __m256 diff = _mm256_sub_ps(vec1_chunk, vec2_chunk);
                    __m256 sq_diff = _mm256_mul_ps(diff, diff);
                    __m256 sum_vec = _mm256_hadd_ps(sq_diff, sq_diff); // horizontal add
                    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);        // second hadd for reduction
                    float temp[8];
                    _mm256_storeu_ps(temp, sum_vec);
                    sum += temp[0] + temp[4];
                }
            #elif defined(__ARM_NEON)
                for (k = 0; k <= d - 4; k += 4) {
                    float32x4_t vec1_chunk = vld1q_f32(&vec1[k]);
                    float32x4_t vec2_chunk = vld1q_f32(&vec2[k]);
                    float32x4_t diff = vsubq_f32(vec1_chunk, vec2_chunk);
                    float32x4_t sq_diff = vmulq_f32(diff, diff);
                    float32x2_t sum_vec = vpadd_f32(vget_low_f32(sq_diff), vget_high_f32(sq_diff));
                    sum += vget_lane_f32(sum_vec, 0) + vget_lane_f32(sum_vec, 1);
                }
            #endif
            // handle remaining elements
            while (k < d) {
                float diff = vec1[k] - vec2[k];
                sum += diff * diff;
                ++k;
            }
        #endif
        return std::sqrt(sum);
    }

    float* l2_distance_optimised(const float* vec, size_t n, size_t d, const float* vec_comp, size_t vec_comp_n) {
        auto* distances = new float[n * vec_comp_n];
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < vec_comp_n; ++j) {
                distances[i * vec_comp_n + j] = l2_distance_optimised_single(vec + i * d, vec_comp + j * d, d);
            }
        }
        return distances;
    }

} // namespace mvdb