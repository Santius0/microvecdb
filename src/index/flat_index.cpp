#include "flat_index.h"
#include "distances.h"

namespace mvdb {

    void FlatIndex::serialize(std::ostream &out) const {

    }

    void FlatIndex::deserialize(std::istream &in) {

    }

    FlatIndex::FlatIndex(const std::string& index_path, const uint64_t& dims):
    Index(index_path, dims, IndexType::FLAT) {}

    void FlatIndex::init(){

    }

    std::vector<uint64_t> FlatIndex::add(const size_t& n, const float* data) const {
        return {};
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