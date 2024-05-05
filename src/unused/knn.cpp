#include "unused/knn.h"
#include "unused/distances.h"
#include <vector>
#include <queue>
#include <omp.h> // OpenMP

#include <iostream>

namespace mvdb {

    struct IndexDistanceMap {
        idx_t vector_index;
        value_t vector_distance;
        IndexDistanceMap(idx_t vector_index, value_t vector_distance) : vector_index(vector_index), vector_distance(vector_distance) {}
        IndexDistanceMap() = default;
    };

    struct SmallerIndexDistanceMap {
        bool operator()(const IndexDistanceMap& a, const IndexDistanceMap& b) const {
            return a.vector_distance > b.vector_distance; // if a.vector_distance > b.vector_distance, a is 'smaller'
                                                          // than b since this is a min heap i.e larger number is worse
        }
    };

    void knn(const value_t* v, const idx_t& nv, const idx_t& d, const value_t* q, const idx_t& nq, const idx_t& k, idx_t* ids, value_t* distances, const DISTANCE_METRIC& distance_metric) {
        if(nv == 0)
            throw std::runtime_error(&"knn must have nv i.e number of vectors > 0. nv = " [ nv]);
        if(nq == 0)
            throw std::runtime_error(&"knn must have nq i.e number of query vectors > 0. nq = " [ nq]);
        if(k == 0)
            throw std::runtime_error(&"knn must have k i.e number of nearest neighbors > 0. k = " [ k]);

        // std::priority_queue is not thread safe so give each thread its own min_heap and merge them at the end in one thread
        std::vector<std::priority_queue<IndexDistanceMap, std::vector<IndexDistanceMap>, SmallerIndexDistanceMap>> thread_min_heaps(omp_get_max_threads());
        std::priority_queue<IndexDistanceMap, std::vector<IndexDistanceMap>, SmallerIndexDistanceMap> main_min_heap;

        float distance = 0.0f;
        #pragma omp parallel for collapse(2)
        for (idx_t i = 0; i < nv; ++i) {
            for (idx_t j = 0; j < nq; ++j) {
                switch (distance_metric) {
                    case L2_DISTANCE:
                        distance = l2_distance_naive_single(v + i * d, q + j * d, d);
                        break;
                    default:
                        throw std::runtime_error("invalid distance metric");
                }
                thread_min_heaps[omp_get_thread_num()].emplace(i, distance);
            }
        }

        //merging individual min_heaps in single large min_heap
        for (const auto& thread_min_heap : thread_min_heaps) {
            std::priority_queue<IndexDistanceMap, std::vector<IndexDistanceMap>, SmallerIndexDistanceMap> tmp_queue = thread_min_heap;
            while (!tmp_queue.empty()) {
                main_min_heap.push(tmp_queue.top());
                tmp_queue.pop();
            }
        }

        for(idx_t i = 0; i < k; ++i) {
            if(!main_min_heap.empty()) {
                ids[i] = main_min_heap.top().vector_index;
                distances[i] = main_min_heap.top().vector_distance;
                main_min_heap.pop();
            } else {
                ids[i] = -1;
                distances[i] = -1.0f;
            }
        }
    }

}