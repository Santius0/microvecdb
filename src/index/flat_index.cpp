#include "flat_index.h"
#include "distances.h"
#include "filesystem.h"
#include <iostream>
#include <fstream>

//#include <vector>
//#include <cmath>
//#include <limits>
#include <queue>
#include <utility>
#include <thread>
#include "heap.h"

namespace mvdb {

    IndexType FlatIndex::type() const {
        return FLAT;
    }

    std::ostream& operator<<(std::ostream& os, const FlatIndex& obj){
        return os   << "index_path_: " << obj.index_path_ << std::endl
                    << "dims_: " << obj.dims_ << std::endl
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
//        // TODO: implement metric_type and save it here
        out.write(reinterpret_cast<const char*>(index_->data()), ntotal_ * dims_ * sizeof(value_t));
    }

    // load everything except index_path as that should be known ahead of time to trigger this load in the first place
    void FlatIndex::deserialize(std::istream& in) {
        auto type = static_cast<IndexType>(deserialize_numeric<uint64_t>(in));
        if (type != FLAT) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
        dims_ = deserialize_numeric<idx_t>(in);
        ntotal_ = deserialize_numeric<idx_t>(in);
//        // TODO: implement metric_type and read it here
        index_ = std::make_shared<std::vector<value_t>>(ntotal_ * dims_);
        in.read(reinterpret_cast<char*>(index_->data()), ntotal_ * dims_ * sizeof(value_t));
    }

    FlatIndex::FlatIndex(const std::string& index_path, const idx_t& dims): Index(index_path, dims) {}

    void FlatIndex::save() const {
        std::ofstream out_file(index_path_, std::ios::binary);
        serialize(out_file);
        out_file.close();
    }

    void FlatIndex::load() {
        std::ifstream in_file(index_path_, std::ios::binary);
        deserialize(in_file);
        in_file.close();
    }

    void FlatIndex::open() {
        if(!is_open_) {
            if(fs::exists(index_path_.c_str())) {
                if (fs::is_directory(index_path_.c_str()))
                    throw std::runtime_error("'" + index_path_ + "' already exists as a directory");
                load();
            }
            else {
                index_ = std::make_shared<std::vector<value_t>>(); // initialise internal index
                save();
            }
            is_open_ = true;
        } else
            throw std::runtime_error("index already open with at '" + index_path_ + "'");
    }

    void FlatIndex::close(bool force) {
        if(is_open_) {
            // insert any cleanup code that should run before object deletion here
            // TODO: if something is running on another thread wait for it to finish first then close, unless force == true
            std::cout << "closing '" << index_path_ << "'..." << std::endl;
            index_.reset();
            ntotal_ = 0;
            dims_ = 0;
            is_open_ = false;
        } else
            throw std::runtime_error("index at '" + index_path_ + "'  not open");
    }

    bool FlatIndex::add(const mvdb::idx_t &n, mvdb::value_t *data, mvdb::idx_t *ids) {
        if (n <= 0 || !data) return false;
        if(!is_open_) throw std::runtime_error("FlatIndex not open. Use index->open() before running add");
        if(!index_) throw std::runtime_error("index_ == nullptr. Use index->open() before running add");

        try {
            index_->reserve(index_->size() + n * dims_);
            for (idx_t i = 0; i < n; ++i) {
                for (idx_t d = 0; d < dims_; ++d) {
                    index_->push_back(data[i * dims_ + d]);
                }
                ids[i] = ++ntotal_;
            }
            if (ntotal_ != (index_->size() / dims_))
                std::cerr << "ERROR: ntotal_ != (index_->size()/dims_) => " << ntotal_ << " != " << index_->size() / dims_ << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error adding data to index: " << e.what() << std::endl;
            return false;
        }
        // The actual parallelization of this process would require more sophisticated approaches,
        // such as using thread pools, SIMD instructions, or GPU acceleration,
        // which are beyond the scope of this basic implementation.
        save();
        return true;
    }

    bool FlatIndex::remove(const mvdb::idx_t &n, const mvdb::idx_t *ids) {
        return false;
    }

    void FlatIndex::search(const idx_t &nq, value_t *queries, idx_t *ids, mvdb::value_t *distances, const idx_t &k) const {
        if(!is_open_) throw std::runtime_error("FlatIndex not open. Use index->open() before running add");

        // For each query vector
        for (idx_t i = 0; i < nq; ++i) {
            // Use a max heap to keep track of the k nearest neighbors found so far
//            MinHeap<std::pair<value_t, idx_t>> heap;
            std::priority_queue<std::pair<value_t, idx_t>> heap;

            // Pointer to the current query vector
            const value_t* query = queries + i * dims_;

            // Compute distance to each vector in the index
            for (idx_t j = 0; j < ntotal_; ++j) {
                const value_t* index_vector = index_->data() + j * dims_;
                value_t dist = l2_distance(query, index_vector, dims_);

                // If the heap is not full yet or the current distance is smaller than the largest distance in the heap
                if (heap.size() < k || dist < heap.top().first) {
                    if (heap.size() == k) {
                        heap.pop(); // Remove the farthest neighbor
                    }
                    heap.push(std::pair<value_t, idx_t>(dist, j)); // Add the current vector
                }
            }

            // Extract the results from the heap into the output arrays
            idx_t j = 0;
            if(heap.size() < k){
                idx_t diff = k - heap.size();
                while(diff > 0){
                    ids[(nq * k) - 1 - (i * k + j)] = -1;
                    distances[(nq * k) - 1 - (i * k + j)] = -1.0f;
                    --diff;
                    ++j;
                }
            }
            while(j < k) {
                if(!heap.empty()) {
                    // Since we're using a max heap, we need to extract results in reverse order
                    auto &top = heap.top();
                    ids[(nq * k) - 1 - (i * k + j)] = top.second;
                    distances[(nq * k) - 1 - (i * k + j)] = top.first;
                    heap.pop();
                }
                ++j;
            }
        }
    }

    value_t* FlatIndex::get(mvdb::idx_t &n, mvdb::idx_t *keys) const {
        return index_->data();
    }

    bool FlatIndex::is_open() const {
        return is_open_;
    }

    idx_t FlatIndex::dims() const {
        return dims_;
    }

    idx_t FlatIndex::ntotal() const {
        return ntotal_;
    }

    std::vector<value_t> FlatIndex::index() const {
        return *index_;
    }
}