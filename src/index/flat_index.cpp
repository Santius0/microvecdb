#include "flat_index.h"
#include "knn.h"
#include "distances.h"
#include "filesystem.h"
#include "constants.h"


#include <iostream>
#include <fstream>
#include <omp.h>

//#include <vector>
//#include <cmath>
//#include <limits>

namespace mvdb {

    IndexType FlatIndex::type() const {
        return FLAT;
    }

    std::ostream& operator<<(std::ostream& os, const FlatIndex& obj){
        return os   << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.type() << std::endl;
    }

    std::ostream& operator<<(std::ostream& os, const FlatIndex* obj){
        return os << "*(" << *obj << ")";
    }

    // save everything except index_path_ as to load the index you'll need to know the path ahead of the load anyway
    void FlatIndex::serialize(std::ostream& out) const {
        serialize_numeric<uint64_t>(out, static_cast<uint64_t>(type()));
        serialize_numeric<uint64_t>(out, dims_);
        serialize_numeric(out, ntotal_);
        out.write(reinterpret_cast<const char*>(index_->data()), ntotal_ * dims_ * sizeof(value_t));
    }

    // load everything except index_path as that should be known ahead of time to trigger this load in the first place
    void FlatIndex::deserialize(std::istream& in) {
        auto type = static_cast<IndexType>(deserialize_numeric<uint64_t>(in));
        if (type != FLAT) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
        dims_ = deserialize_numeric<idx_t>(in);
        ntotal_ = deserialize_numeric<idx_t>(in);
        index_->reserve(ntotal_ * dims_);
        in.read(reinterpret_cast<char*>(index_->data()), ntotal_ * dims_ * sizeof(value_t));
    }

    FlatIndex::FlatIndex(const idx_t& dims, const std::string& path): Index(dims, path) {
        index_ = std::make_shared<std::vector<value_t>>();
        if(!path.empty() && fs::exists(path) && !fs::is_directory(path)) load(path);
        else index_->reserve(100);
    }

    void FlatIndex::save(const std::string& path) const {
        std::ofstream out_file(path, std::ios::binary);
        serialize(out_file);
        out_file.close();
    }

    void FlatIndex::load(const std::string& path) {
        std::ifstream in_file(path, std::ios::binary);
        deserialize(in_file);
        in_file.close();
    }

    bool FlatIndex::add(const mvdb::idx_t &n, mvdb::value_t *data, mvdb::idx_t *ids = nullptr) {
        if (n <= 0 || !data) return false;
        if(!index_) throw std::runtime_error("index_ == nullptr. Use index->open() before running add");

        try {
            index_->reserve(index_->size() + n * dims_);    // ensure we have enough space to hold new elements
            for (idx_t i = 0; i < n; ++i) {
                for (idx_t j = 0; j < dims_; ++j) {
                    index_->push_back(data[i * dims_ + j]);
                }
                ntotal_++;
                if(ids) ids[i] = ntotal_;
            }
            if (ntotal_ != (index_->size() / dims_))    // num vectors must be equal to num elements/num dimensions
                std::cerr << "ERROR: ntotal_ != (index_->size()/dims_) => " << ntotal_ << " != " << index_->size() / dims_ << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error adding data to index: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    bool FlatIndex::remove(const mvdb::idx_t &n, const mvdb::idx_t *ids) {
        return false;
    }

    void FlatIndex::search(const idx_t &nq, value_t *queries, idx_t *ids, value_t *distances, const idx_t &k, const DISTANCE_METRIC& distance_metric) const {
        if(!index_ || !index_->data()) throw std::runtime_error("index_ not properly initialized. index_ == nullptr || index_data == nullptr");
        knn(index_->data(), ntotal_, dims_, queries, nq, k, ids, distances, distance_metric);
    }

//    void FlatIndex::search(const idx_t &nq, value_t *queries, idx_t *ids, mvdb::value_t *distances, const idx_t &k) const {
//        // For each query vector
//        for (idx_t i = 0; i < nq; ++i) {
//            // Use a max heap to keep track of the k nearest neighbors found so far
//            // MinHeap<std::pair<value_t, idx_t>> heap;
//            std::priority_queue<std::pair<value_t, idx_t>> heap;
//
//            // Pointer to the current query vector
//            const value_t* query = queries + i * dims_;
//
//            // Compute distance to each vector in the index
//            for (idx_t j = 0; j < ntotal_; ++j) {
//                const value_t* index_vector = index_->data() + j * dims_;
//                value_t dist = squared_l2_distance_naive(query, index_vector, dims_);
//                dist = std::sqrt(dist);
//                // If the heap is not full yet or the current distance is smaller than the largest distance in the heap
//                if (heap.size() < k || dist < heap.top().first) {
//                    if (heap.size() == k) {
//                        heap.pop(); // Remove the farthest neighbor
//                    }
//                    heap.push(std::pair<value_t, idx_t>(dist, j)); // Add the current vector
//                }
//            }
//
//            // Extract the results from the heap into the output arrays
//            idx_t j = 0;
//            if(heap.size() < k){
//                idx_t diff = k - heap.size();
//                while(diff > 0){
//                    ids[(nq * k) - 1 - (i * k + j)] = -1;
//                    distances[(nq * k) - 1 - (i * k + j)] = -1.0f;
//                    --diff;
//                    ++j;
//                }
//            }
//            while(j < k) {
//                if(!heap.empty()) {
//                    // Since we're using a max heap, we need to extract results in reverse order
//                    auto &top = heap.top();
//                    ids[(nq * k) - 1 - (i * k + j)] = top.second;
//                    distances[(nq * k) - 1 - (i * k + j)] = top.first;
//                    heap.pop();
//                }
//                ++j;
//            }
//        }
//    }

    value_t* FlatIndex::get(idx_t &n, idx_t *keys) const {
        if(n <= 0 || !keys) return nullptr;
        auto *vecs = new value_t[n * dims_];
        #pragma omp parallel for collapse(2)
        for(idx_t i = 0; i < n; ++i) {
            for (idx_t j = 0; j < dims_; ++j) {
                vecs[i * dims_ + j] = index_->at(keys[i] * dims_ + j);
            }
        }
        return vecs;
    }

    value_t* FlatIndex::get_all() const {
        return index_->data();
    }

    idx_t FlatIndex::dims() const {
        return dims_;
    }

    idx_t FlatIndex::ntotal() const {
        return ntotal_;
    }

    value_t* FlatIndex::index() const {
        return index_->data();
    }
}