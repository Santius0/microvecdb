#include "flat_index.h"
#include "distances.h"

namespace mvdb {
    FlatIndex::FlatIndex(std::string obj_name, const DBObjectType& obj_type, std::string obj_col_name,
                         const mvdb::IndexType &idx_type, const uint64_t &idx_dims):
                         Index(std::move(obj_name), obj_type, std::move(obj_col_name), idx_type, idx_dims) {

    }

    void FlatIndex::serialize(std::ostream &out) const {

    }

    void FlatIndex::deserialize(std::istream &in) {

    }

    void FlatIndex::add(const mvdb::vec_count_t& n, const float* __restrict x) {
        #pragma omp parallel for if (n > 100)
        for(vec_count_t i = 0; i < n; i++)
            data_.emplace_back(x[i]);
    }

    void FlatIndex::search(const vec_count_t& k, const vec_count_t& n, const float* __restrict x, const pkey_t* __restrict pkeys, float* __restrict distances) const {
        if(pkeys == nullptr) pkeys = new pkey_t [k];
        if(distances == nullptr) distances = new float [k];


    }

    std::vector<int> findKNearestNeighbors(const std::vector<float>& query, const std::vector<std::vector<float>>& dataset, int k) {
        // Use a max heap to keep track of the k nearest neighbors found so far
        std::priority_queue<std::pair<float, int>> neighbors;

        for (size_t i = 0; i < dataset.size(); ++i) {
            float distance = L2_distance(query, dataset[i]);
            if (neighbors.size() < k) {
                neighbors.push(std::make_pair(distance, i));
            } else if (distance < neighbors.top().first) {
                neighbors.pop();
                neighbors.push(std::make_pair(distance, i));
            }
        }

        // Extract the indices of the k nearest neighbors
        std::vector<int> nearestIndices;
        while (!neighbors.empty()) {
            nearestIndices.push_back(neighbors.top().second);
            neighbors.pop();
        }
        return nearestIndices;
    }

}