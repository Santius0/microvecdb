#ifndef VECTOR_collection_H
#define VECTOR_collection_H

#include "kv_store.h"
#include "vector_index.h"
#include "vectorizer.h"

#include <memory>

namespace mvdb {

    class VectorCollection {
        std::unique_ptr<KvStore> kv_store_;
        std::unique_ptr<VectorIndex> vector_index_;
        std::unique_ptr<Vectorizer> vectorizer;
        std::string name;
        std::string path;
    public:
        VectorCollection(std::string name, std::string path);
        ~VectorCollection() = default;
        bool add(char* data, float* vector = nullptr);
        bool remove(uint64_t key);
        uint64_t* search(float* vectors, size_t num_vectors);
    };

}
#endif // VECTOR_collection_H
