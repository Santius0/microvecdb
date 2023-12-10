#include "microvecdb_c.h"

    // mvdb::VectorCollectionMetadata* create_new_vector_collection_metadata();

mvdb::KvStoreMetadata* KvStoreMetadata_new(const std::string& dataDirectoryPath, const bool& create_if_missing) {
    rocksdb::Options options;
    options.create_if_missing = create_if_missing;
    return new mvdb::KvStoreMetadata(dataDirectoryPath, options);
}

void KvStoreMetadata_delete(const mvdb::KvStoreMetadata* obj) {
    delete obj;
}

void print_kv_store_metadata(const void* kv_store_metadata) {
    std::cout << kv_store_metadata;
}