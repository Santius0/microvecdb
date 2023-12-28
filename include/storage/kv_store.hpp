#ifndef KV_STORE_H
#define KV_STORE_H

#include "constants.hpp"
#include "serializable.hpp"
#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include <rocksdb/db.h>
#include <rocksdb/options.h>

namespace mvdb {

    class KvStore final: Serializable {
        std::unique_ptr<rocksdb::DB> db_{};
        std::string data_dir_path_{};
        rocksdb::Options options_{};

        bool is_open_ = false;
        friend class VectorDB;
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        // Constructor
        explicit KvStore(std::string  data_dir_path, const bool& create_if_missing = true, const bool& open = false);

        // Destructor
        ~KvStore() override = default;

        // Non-copyable and non-movable
        KvStore(const KvStore&) = delete;
        KvStore& operator=(const KvStore&) = delete;

        void open(const std::string& path = "");

        void close();

        // Function to add data with a key-value pair
        // Returns true on success, false on failure
         [[nodiscard]] bool put(const std::string& key, const std::string& value) const;

         [[nodiscard]] int64_t putAutoKey(const std::string& value) const;

        // Function to batch add data with key-value pairs
        // Returns true on success, false on failure
         bool putMany(const std::vector<std::pair<std::string, std::string>>& pairs) const;

         bool putManyAutoKey(const std::vector<std::string>& values) const;

        // Function to retrieve data by key
        // Returns the value or empty string if the key does not exist
         [[nodiscard]] std::string get(const std::string& key) const;

        // Function to remove data using a key
        // Returns true on success, false if the key does not exist
         [[nodiscard]] bool remove(const std::string& key) const;

         [[nodiscard]] bool is_open() const;

    };

}
#endif // KV_STORE_H
