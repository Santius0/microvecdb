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
        save();
    }

    void FaissFlatIndex::deserialize(std::istream& in) {
        Index::deserialize(in);
    }

    FaissFlatIndex::FaissFlatIndex(const std::string& index_path, const uint64_t& dims):
    Index(index_path, dims, IndexType::FAISS_FLAT) {}

    void FaissFlatIndex::init(){
        faiss_index_ = std::make_unique<faiss::IndexFlatL2>(dims_);
    }

    bool FaissFlatIndex::add(const size_t& n, const float* data, uint64_t* keys) const {
        try {
            for(int i = 0; i < n; i++)
                keys[i] = faiss_index_->ntotal + i;
            faiss_index_->add(static_cast<long>(n), data);
            save();
        } catch (const std::exception& e) {
            std::cerr << "Error adding data to index: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    bool FaissFlatIndex::remove(const size_t& n, const faiss::IDSelector& ids) const {
        try {
            faiss_index_->remove_ids(ids);
            save();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error removing data from index: " << e.what() << std::endl;
            return false;
        }
    }

    void FaissFlatIndex::save() const {
        if(!is_open_)
            faiss::write_index(faiss_index_.get(), index_path_.c_str());
    }

    void FaissFlatIndex::load() {
        faiss_index_.reset(faiss::read_index(index_path_.c_str()));
    }

    void FaissFlatIndex::search(const int& n, float* query, int64_t* ids, float* distances, const long& k) const{
        faiss_index_->search(static_cast<long>(n/dims_), query, k, distances, ids);
    }

    void FaissFlatIndex::close() {
        faiss_index_->reset();
        is_open_ = false;
    }

    faiss::Index* FaissFlatIndex::faiss_index(){
        return faiss_index_.get();
    }

} // namespace mvdb
