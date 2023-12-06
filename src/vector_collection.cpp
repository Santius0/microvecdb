#include "vector_collection.hpp"
#include "constants.hpp"
#include "utils.hpp"

#include <filesystem>

namespace mvdb {
    VectorCollection::VectorCollection(const std::string& name, const std::string& path, const uint64_t& dimensions,
        const std::string& model): name(name), path(trim(path)), dimensions(dimensions),
    model(model) {
        meta_path = path + std::filesystem::path::preferred_separator + name + COLLECTION_META_EXT;
        kv_path = path + std::filesystem::path::preferred_separator + name + KV_STORE_EXT;
        index_path = path + std::filesystem::path::preferred_separator + name + INDEX_EXT;
        index_meta_path = path + std::filesystem::path::preferred_separator + name + INDEX_META_EXT;
        vectorizer_ = std::make_unique<Vectorizer>(this->model, this->dimensions);
        kv_store_ = std::make_unique<KvStore>(this->kv_path, false, false);
        vector_index_ = std::make_unique<VectorIndex>(this->name, this->path, VectorIndexType::FLAT, this->dimensions);
    }

    VectorCollection* VectorCollection::load(const std::string& path) {
        // return new VectorCollection();
        return nullptr;
    }

}