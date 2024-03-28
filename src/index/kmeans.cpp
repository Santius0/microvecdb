#include "kmeans.h"
#include "distances.h"

#include <iostream>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <immintrin.h> // AVX intrinsics

#include "constants.h"


namespace mvdb {

    // vec = single vector to compare to centroids
    // centroids = array of centroid
    // k = number of centroids
    // d = dimensionality of each centroid
    size_t nearest_centroid(const float* vec, const float* centroids, const size_t& k, const size_t& d, const DISTANCE_METRIC& distance_metric){
        size_t nearest_centroid_idx = 0;
        float min_dist = std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < k; ++j) {
            float dist;
            switch (distance_metric) {
                case MAHALANOBIS_DISTANCE:
                    dist = l2_distance_optimised_single(&vec[0], &centroids[j * d], d);
                    break;
                    // TODO: implement maha dist
                default:
                    dist = l2_distance_optimised_single(&vec[0], &centroids[j * d], d);
            }
            if (dist < min_dist) {
                min_dist = dist;
                nearest_centroid_idx = j;
            }
        }
        return nearest_centroid_idx;
    }

    float* kmeans(const float* vec, const size_t& n, const size_t& d, const size_t& k, const int& max_iterations, const DISTANCE_METRIC& distance_metric) {

        // initialising k centroids with random values
        srand(time(nullptr));
        auto* centroids = new float[k * d];
        for (size_t i = 0; i < k; ++i) {
            size_t randIndex = rand() % n;
            for (size_t j = 0; j < d; ++j) {
                centroids[i * d + j] = vec[randIndex * d + j];
            }
        }

        auto* assignments = new size_t[n];
        bool centroids_changed = true;
        size_t iterations = 0;

        while (centroids_changed && iterations < max_iterations) {
            centroids_changed = false;

            // Parallel assignment step
            #pragma omp parallel for shared(vec, centroids, assignments) reduction(|| : centroids_changed)
            for (size_t i = 0; i < n; ++i) {

                // ---------- closest centroid -----------------------------
//                float minDist = std::numeric_limits<float>::infinity();
//                size_t nearestCentroid = 0;
//                for (size_t j = 0; j < k; ++j) {
//                    float dist;
//                    switch (distance_metric) {
//                        case MAHALANOBIS_DISTANCE:
//                            dist = l2_distance_optimised_single(&vec[i * d], &centroids[j * d], d);
//                            // TODO: implement maha dist
//                        default:
//                             dist = l2_distance_naive_single(&vec[i * d], &centroids[j * d], d);
//                    }
//                    if (dist < minDist) {
//                        minDist = dist;
//                        nearestCentroid = j;
//                    }
//                }
                size_t nearestCentroid = nearest_centroid(vec + i * d, centroids, k, d, distance_metric);
                // ---------- closest centroid -----------------------------
                if (assignments[i] != nearestCentroid) {
                    assignments[i] = nearestCentroid;
                    centroids_changed = true;
                }
            }

            // Parallel update step
            #pragma omp parallel for shared(vec, centroids, assignments)
            for (size_t i = 0; i < k; ++i) {
                float* tempCentroid = new float[d]();
                size_t count = 0;
                #pragma omp parallel for reduction(+:tempCentroid[:d], count)
                for (size_t j = 0; j < n; ++j) {
                    if (assignments[j] == i) {
                        for (size_t m = 0; m < d; ++m) {
                            tempCentroid[m] += vec[j * d + m];
                        }
                        count++;
                    }
                }
                if (count > 0) {
                    for (size_t m = 0; m < d; ++m) {
                        centroids[i * d + m] = tempCentroid[m] / count;
                    }
                }
                delete[] tempCentroid;
            }

            iterations++;
        }

        delete[] assignments;
        return centroids;
    }

}