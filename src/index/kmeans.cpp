#include "kmeans.h"
#include "distances.h"
#include <vector>
#include <queue>
#include <omp.h> // OpenMP

namespace mvdb {

    value_t* kmeans(const value_t* v, const idx_t& nv, const idx_t& d, const value_t* q, const idx_t& nq, const idx_t& k, DISTANCE_METRIC distance_metric = L2_DISTANCE) {
        std::priority_queue<value_t, std::vector<value_t>, std::greater<>> min_heap;
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < nv; ++i) {
            for (size_t j = 0; j < nq; ++j) {
                float sq_distance;
                switch (distance_metric) {
                    case MAHALANOBIS_DISTANCE:
                        sq_distance = squared_l2_distance_naive(v + i * d, q + j * d, d);
                        break;
                    default:
                        sq_distance = squared_l2_distance_naive(v + i * d, q + j * d, d);
                }
                min_heap.push(sq_distance);
            }
        }
        auto* res = new value_t[k];
        for(idx_t i = 0; i < k; ++i) {
            res[i] = min_heap.top();
            min_heap.pop();
        }
        return res;
    }

}