#include "faiss_flat_index.h"
#include <iostream>
#include <filesystem>

namespace mvdb {

    std::ostream& operator<<(std::ostream& os, const FaissFlatIndex& obj){
        return os   << "index_path_: " << obj.index_path_ << std::endl
                    << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.index_type_ << std::endl;
    }

    std::ostream& operator<<(std::ostream& os, const FaissFlatIndex* obj){
        return os << "*(" << *obj << ")";
    }

    void FaissFlatIndex::serialize(std::ostream& out) const {
        Index::serialize(out);
    }

    void FaissFlatIndex::deserialize(std::istream& in) {
        Index::deserialize(in);
    }

    FaissFlatIndex::FaissFlatIndex(const std::string& index_path, const uint64_t& dims):
    Index(index_path, dims, IndexType::FAISS_FLAT) {}

    void FaissFlatIndex::init(){
        faiss_index = std::make_unique<faiss::IndexFlatL2>(dims_);
    }

    std::vector<uint64_t> FaissFlatIndex::add(const size_t& n, const float* data) const {
        try {
            std::vector<uint64_t> keys;
            keys.reserve(n);
            for(int i = 0; i < n; i++)
                keys.emplace_back(faiss_index->ntotal + i);
            faiss_index->add(static_cast<long>(n), data);
//            if(ids == nullptr) faiss_index->add(static_cast<long>(n), data);
//            else faiss_index->add_with_ids(static_cast<long>(n), data, ids);
            save();
            return keys;
        } catch (const std::exception& e) {
            std::cerr << "Error adding data to index: " << e.what() << std::endl;
            return {};
        }
    }

    bool FaissFlatIndex::remove(const size_t& n, const faiss::IDSelector& ids) const {
        try {
            faiss_index->remove_ids(ids);
            save();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error removing data from index: " << e.what() << std::endl;
            return false;
        }
    }

    void FaissFlatIndex::save() const {
        faiss::write_index(faiss_index.get(), index_path_.c_str());
    }

    void FaissFlatIndex::load() {
        faiss_index.reset(faiss::read_index(index_path_.c_str()));
    }

    void FaissFlatIndex::search(const std::vector<float>& query, int64_t ids[], float distances[], const long& k) const {
        faiss_index->search(static_cast<long>(query.size()/dims_), query.data(), k, distances, ids);
    }

    void FaissFlatIndex::close() {
        faiss_index->reset();
        is_open_ = false;
    }

} // namespace mvdb
