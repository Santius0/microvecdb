#ifndef KV_STORE_H
#define KV_STORE_H

#include "serializable.hpp"
#include <string>
#include <vector>
#include <memory>
#include <rocksdb/db.h>
#include <rocksdb/options.h>

namespace mvdb {

    class KvStoreMetadata final: public Serializable {
        rocksdb::Options options_;
        std::string dataDirectoryPath;
        friend class Metadata;
        friend class CollectionMetadata;
        friend class VectorCollection;
        friend class KvStore;
    protected:
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
    public:
        KvStoreMetadata() = default;
        explicit KvStoreMetadata(std::string dataDirectoryPath, const rocksdb::Options& options);
        ~KvStoreMetadata() override = default;
    };

    class KvStore {
        std::unique_ptr<rocksdb::DB> db;
        rocksdb::Options options;
    public:
        // Constructor
        explicit KvStore(const KvStoreMetadata& metadata);

        // Destructor
        ~KvStore() = default;

        // Non-copyable and non-movable
        KvStore(const KvStore&) = delete;
        KvStore& operator=(const KvStore&) = delete;

        // Function to add data with a key-value pair
        // Returns true on success, false on failure
        [[nodiscard]] bool put(const std::string& key, const std::string& value) const;

        [[nodiscard]] int64_t putAutoKey(const std::string& value) const;

        // Function to batch add data with key-value pairs
        // Returns true on success, false on failure
        [[nodiscard]] bool putMany(const std::vector<std::pair<std::string, std::string>>& pairs) const;

        [[nodiscard]] bool putManyAutoKey(const std::vector<std::string>& values) const;

        // Function to retrieve data by key
        // Returns the value or empty string if the key does not exist
        [[nodiscard]] std::string get(const std::string& key) const;

        // Function to remove data using a key
        // Returns true on success, false if the key does not exist
        [[nodiscard]] bool remove(const std::string& key) const;

    };

}
#endif // KV_STORE_H
