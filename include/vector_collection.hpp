#ifndef VECTOR_collection_H
#define VECTOR_collection_H

#include "kv_store.hpp"
#include "vector_index.hpp"
#include "vectorizer.hpp"

#include <memory>

namespace mvdb {

    class VectorCollection {
        std::unique_ptr<KvStore> kv_store_;
        std::unique_ptr<VectorIndex> vector_index_;
        std::unique_ptr<Vectorizer> vectorizer_;
        std::string name;
        std::string path;
        uint64_t dimensions;
        std::string model;

        std::string kv_path;
        std::string index_path;
        std::string index_meta_path;
        std::string meta_path;
    public:
        VectorCollection(const std::string& name, const std::string& path, const uint64_t& dimensions,
            const std::string& model);
        ~VectorCollection() = default;
        static VectorCollection* load(const std::string& path);
        bool add(char* data, float* vector = nullptr);
        bool remove(uint64_t key);
        uint64_t* search(float* vectors, size_t num_vectors);
    };

}
#endif // VECTOR_collection_H
