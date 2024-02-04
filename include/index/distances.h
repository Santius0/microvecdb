#ifndef MICROVECDB_DISTANCES_H
#define MICROVECDB_DISTANCES_H

#include "constants.h"
//#include <iostream>
//#include <vector>
#include <cmath>
#include <omp.h>
//#include <queue>
//#include <utility>

namespace mvdb {

    // computes the l2_distance between two vectors a and b, both with dims dimensions.
    // computed distance returned as a single value_t.
    inline value_t l2_distance(const value_t* a, const value_t* b, idx_t dims) {
        value_t distance = 0;
        #pragma omp simd reduction(+:distance)
        for (size_t d = 0; d < dims; ++d) {
            value_t diff = a[d] - b[d];
            distance += diff * diff;
        }
        return std::sqrt(distance);
    }

    // computes the l2_distance between n vectors, a_0, a_1, ..., a_n and b, all with dims dimensions.
    // the respective distances of [a_0, b], [a_1, b], ... , [a_n, b] are returned as a value_t*.
    inline value_t* l2_distance(const value_t* a, const value_t* b, idx_t n, idx_t dims) {
        auto* distances = new value_t[n];
        #pragma omp parallel for // default(none) shared(a, b, n, dims) private(i)
        for (idx_t i = 0; i < n; ++i) {
            value_t distance = 0;
            // Pointer to the start of the i-th vector in a
            const value_t* a_i = a + i * dims;

            // Enable SIMD vectorization for the inner loop
            #pragma omp simd reduction(+:distance)
            for (idx_t d = 0; d < dims; ++d) {
                value_t diff = a_i[d] - b[d];
                distance += diff * diff;
            }
            distances[i] = std::sqrt(distance);
        }
        return distances;
    }

//// Function to find the k nearest neighbors
//    inline std::vector<uint64_t> knn(const std::vector<float> &query, const std::vector<std::vector<float>> &dataset, int k) {
//        // Use a max heap to keep track of the k nearest neighbors found so far
//        std::priority_queue<std::pair<float, int>> neighbors;
//
//        for (size_t i = 0; i < dataset.size(); ++i) {
//            float distance = l2_distance(query, dataset[i]);
//            if (neighbors.size() < k) {
//                neighbors.emplace(distance, i);
//            } else if (distance < neighbors.top().first) {
//                neighbors.pop();
//                neighbors.emplace(distance, i);
//            }
//        }
//
//        // Extract the indices of the k nearest neighbors
//        std::vector<uint64_t> nearestIndices;
//        while (!neighbors.empty()) {
//            nearestIndices.push_back(neighbors.top().second);
//            neighbors.pop();
//        }
//        return nearestIndices;
//    }
//
//    template<typename T, typename D>
//    inline std::vector<D> L2_distance_vectorized(const T* query, const T* datasetVector, idx_t nq, idx_t dims) {
//        std::vector<D> distances(nq);
//        for (idx_t i = 0; i < nq; ++i) {
//            D sum = 0;
//            for (idx_t j = 0; j < dims; ++j) {
//                D diff = static_cast<D>(query[i * dims + j] - datasetVector[j]);
//                sum += diff * diff;
//            }
//            distances[i] = std::sqrt(sum);
//        }
//        return distances;
//    }
}
#endif //MICROVECDB_DISTANCES_H
