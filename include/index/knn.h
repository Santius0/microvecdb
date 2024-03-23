#ifndef MICROVECDB_KNN_H
#define MICROVECDB_KNN_H

#include "constants.h"

namespace mvdb {

    void knn(const value_t* v, const idx_t& nv, const idx_t& d, const value_t* q, const idx_t& nq, const idx_t& k, idx_t* ids, value_t* distances, const DISTANCE_METRIC& distance_metric = L2_DISTANCE);
}

#endif //MICROVECDB_KNN_H
