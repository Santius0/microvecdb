#include "flat_index.h"
#include "distances.h"

namespace mvdb {

    std::ostream& operator<<(std::ostream& os, const FlatIndex& obj){
        return os   << "index_path_: " << obj.index_path_ << std::endl
                    << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.index_type_ << std::endl;
    }

    std::ostream& operator<<(std::ostream& os, const FlatIndex* obj){
        return os << "*(" << *obj << ")";
    }

    void FlatIndex::serialize(std::ostream &out) const {
        Index::serialize(out);
    }

    void FlatIndex::deserialize(std::istream &in) {
        Index::deserialize(in);
    }

    FlatIndex::FlatIndex(const std::string& index_path, const uint64_t& dims):
    Index(index_path, dims, IndexType::FLAT) {}

    void FlatIndex::init(){}

    void FlatIndex::add(const size_t& n, const value_t* data) const {
        this->index_.push
    };

    [[nodiscard]] bool FlatIndex::remove(const size_t& n, const faiss::IDSelector& ids) const {
        return true;
    }
    void FlatIndex::save() const {

    }

    void FlatIndex::load() {

    }

    void FlatIndex::close() {

    };

    void FlatIndex::search(const std::vector<float>& query, int64_t ids[], float distances[], const long& k) const {

    }
}