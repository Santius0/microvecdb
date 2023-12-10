#include "vector_collection.hpp"
#include "utils.hpp"
#include <filesystem>
#include <utility>

namespace mvdb {

    VectorCollectionMetadata::VectorCollectionMetadata(std::string name, std::string collectionFilePath, const size_t& recordCount,
        const KvStoreMetadata& kv_store_metadata, const VectorIndexMetadata& vector_index_metadata,
        const VectorizerMetadata& vectorizer_metadata):
    name(std::move(name)), collectionFilePath(std::move(collectionFilePath)), recordCount(recordCount) {
        createdTimestamp = getCurrentTimeStamp();
        modifiedTimestamp = createdTimestamp;

        kv_store_metadata_ = kv_store_metadata;
        vector_index_metadata_ = vector_index_metadata;
        vectorizer_metadata_ = vectorizer_metadata;
    }

    void VectorCollectionMetadata::serialize(std::ostream& out) const {
        serializeString(out, name);
        serializeString(out, collectionFilePath);
        serializeUInt64T(out, recordCount);
        serializeString(out, createdTimestamp);
        serializeString(out, modifiedTimestamp);
        kv_store_metadata_.serialize(out);
        vector_index_metadata_.serialize(out);
        vectorizer_metadata_.serialize(out);
    }

    void VectorCollectionMetadata::deserialize(std::istream& in) {
        name = deserializeString(in);
        collectionFilePath = deserializeString(in);
        recordCount = deserializeUInt64T(in);
        createdTimestamp = deserializeString(in);
        modifiedTimestamp = deserializeString(in);
        kv_store_metadata_.deserialize(in);
        vector_index_metadata_.deserialize(in);
        vectorizer_metadata_.deserialize(in);
    }

    VectorCollection::VectorCollection(const VectorCollectionMetadata& metadata) {
        if(!std::filesystem::exists(metadata.collectionFilePath)) std::filesystem::create_directory(metadata.collectionFilePath);
        kv_store_ = std::make_unique<KvStore>(metadata.kv_store_metadata_);
        vector_index_ = std::make_unique<VectorIndex>(metadata.vector_index_metadata_);
        vectorizer_ = std::make_unique<Vectorizer>(metadata.vectorizer_metadata_);
    }

    bool VectorCollection::add_data(const std::string& data) const {
        fasttext::Vector vec = vectorizer_->get_word_vector(data);
        const std::vector<uint64_t> keys = vector_index_->add(1, vec.data());
        return kv_store_->put(std::to_string(keys[0]), data);
    }

    SearchResult VectorCollection::search_with_vector(const std::vector<float>& query, const long& k, const bool& ret_data) const {
        auto *ids = new int64_t[k];
        auto *distances = new float[k];
        auto *data = new std::string[k];
        vector_index_->search(query, ids, distances, k);
        if(ret_data) for(int i = 0 ; i < k; i++) data[i] = kv_store_->get(std::to_string(ids[i]));
        return {ids, distances, data, k};
    }

    SearchResult VectorCollection::search(const std::string& data, const long& k, const bool& ret_data) const {
        fasttext::Vector query = vectorizer_->get_word_vector(data);
        return search_with_vector(query.get_data_(), k, ret_data);
    }

}