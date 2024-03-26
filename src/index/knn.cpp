#include "knn.h"
#include "distances.h"
#include <vector>
#include <queue>
#include <omp.h> // OpenMP

#include <iostream>

namespace mvdb {

    struct IndexDistanceMap {
        idx_t vector_index;
        value_t vector_distance;
        IndexDistanceMap(idx_t vector_index, value_t vector_distance) : vector_index(vector_index), vector_distance(vector_distance) {}
        IndexDistanceMap(){}
    };

    struct SmallerIndexDistanceMap {
        bool operator()(const IndexDistanceMap& a, const IndexDistanceMap& b) const {
            return a.vector_distance > b.vector_distance; // For min heap
        }
    };

    void knn(const value_t* v, const idx_t& nv, const idx_t& d, const value_t* q, const idx_t& nq, const idx_t& k, idx_t* ids, value_t* distances, const DISTANCE_METRIC& distance_metric) {
        std::priority_queue<IndexDistanceMap, std::vector<IndexDistanceMap>, SmallerIndexDistanceMap> min_heap;
        float distance;
        #pragma omp parallel for collapse(2)
        for (idx_t i = 0; i < nv; ++i) {
            for (idx_t j = 0; j < nq; ++j) {
                switch (distance_metric) {
                    case MAHALANOBIS_DISTANCE:
                        distance = l2_distance_naive_single(v + i * d, q + j * d, d);
                        break;
                    default:
                        distance = l2_distance_naive_single(v + i * d, q + j * d, d);
                }
                IndexDistanceMap idx_map = IndexDistanceMap(i, distance);
                min_heap.push(idx_map);
            }
        }
        for(idx_t i = 0; i < k; ++i) {
            ids[i] = min_heap.top().vector_index;
            distances[i] = min_heap.top().vector_distance;
            std::cout << ids[i] << " => " << distances[i] << std::endl;
            min_heap.pop();
        }
    }

}