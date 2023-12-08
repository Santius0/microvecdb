#ifndef VECTOR_collection_H
#define VECTOR_collection_H

#include "kv_store.hpp"
#include "vector_index.hpp"
#include "vectorizer.hpp"
#include "serializable.h"
#include <memory>
#include <ostream>
#include <string>

namespace mvdb {

    class CollectionMetadata final: public Serializable {
        friend class Metadata;
    protected:
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
    public:
        std::string name;
        std::string collectionFilePath;
        size_t recordCount{};
        std::string createdTimestamp;
        std::string modifiedTimestamp;
        KvStoreMetadata kv_store_metadata_;
        VectorIndexMetadata vector_index_metadata_;
        VectorizerMetadata vectorizer_metadata_;
        CollectionMetadata() = default;
        CollectionMetadata(std::string name, std::string collectionFilePath, const size_t& recordCount,
        const KvStoreMetadata& kv_store_metadata, const VectorIndexMetadata& vector_index_metadata,
        const VectorizerMetadata& vectorizer_metadata);
        ~CollectionMetadata() override = default;
        friend std::ostream& operator<<(std::ostream& os, const CollectionMetadata& obj) {
            return os
                   << "name: " << obj.name
                   << " recordCount: " << obj.recordCount
                   << " createdTimestamp: " << obj.createdTimestamp
                   << " modifiedTimestamp: " << obj.modifiedTimestamp;
        }
    };

    class SearchResult {
        friend std::ostream& operator<<(std::ostream& os, const SearchResult& obj) {
            for(int i = 0; i < obj.size_; i++)
                os << "id: " << obj.ids_.get()[i]
                   << "\tdistance: "
                   << obj.distances_.get()[i]
                   // << "\tdata: " << obj.data[i]
                   << std::endl
                ;
            return os;
        }

        friend std::ostream& operator<<(std::ostream& os, const SearchResult* obj) {
            return os << "*(" << *obj << ")";
        }

    public:
        std::unique_ptr<int64_t> ids_;
        std::unique_ptr<float> distances_;
        std::unique_ptr<std::string> data_;
        const long size_;
        SearchResult(int64_t* ids, float* distances, std::string* data, const long& size):
        size_(size) {
            ids_.reset(ids);
            distances_.reset(distances);
            data_.reset(data);
        }
        ~SearchResult() = default;
    };

    class VectorCollection {
        std::unique_ptr<KvStore> kv_store_;
        std::unique_ptr<VectorIndex> vector_index_;
        std::unique_ptr<Vectorizer> vectorizer_;
    public:
        VectorCollection() = default;
        explicit VectorCollection(const CollectionMetadata& metadata);
        ~VectorCollection() = default;
        bool add_data(const std::string& data) const;
        SearchResult search_with_vector(const std::vector<float>& query, const long& k = 5, const bool& ret_data = false) const;
        SearchResult search(const std::string& data, const long& k = 5, const bool& ret_data = false) const;
        // bool remove(uint64_t key);
        // uint64_t* search(float* vectors, size_t num_vectors);
    };

}
#endif // VECTOR_collection_H
