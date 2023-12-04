#ifndef KV_STORE_H
#define KV_STORE_H

#include <string>
#include <vector>
#include <memory>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>

namespace mvdb {

    class KvStore {
        std::unique_ptr<rocksdb::DB> db;
        rocksdb::Options options;
    public:
        // Constructor
        KvStore(const std::string& path, bool createNew, bool overwrite);

        // Destructor
        ~KvStore() = default;

        // Non-copyable and non-movable
        KvStore(const KvStore&) = delete;
        KvStore& operator=(const KvStore&) = delete;

        // Function to add data with a key-value pair
        // Returns true on success, false on failure
        [[nodiscard]] bool put(const std::string& key, const std::string& value) const;

        // Function to batch add data with key-value pairs
        // Returns true on success, false on failure
        [[nodiscard]] bool putMany(const std::vector<std::pair<std::string, std::string>>& pairs) const;

        // Function to retrieve data by key
        // Returns the value or empty string if the key does not exist
        [[nodiscard]] std::string get(const std::string& key) const;

        // Function to remove data using a key
        // Returns true on success, false if the key does not exist
        [[nodiscard]] bool remove(const std::string& key) const;

    };

}
#endif // KV_STORE_H
