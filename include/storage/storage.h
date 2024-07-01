#ifndef MICROVECDB_STORAGE_H
#define MICROVECDB_STORAGE_H

#include "serializable.h"
#include <rocksdb/db.h>

namespace mvdb {
    class Storage final : Serializable {
        std::unique_ptr<rocksdb::DB> db_{};
        std::string data_dir_path_{};
        rocksdb::Options options_{};
        bool is_open_ = false;
        friend std::ostream& operator<<(std::ostream& os, const Storage& obj);
        friend std::ostream& operator<<(std::ostream& os, const Storage* obj);
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        explicit Storage(std::string  data_dir_path, const bool& create_if_missing = true, const bool& open = false);
        ~Storage() override = default;
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        void open(const std::string& path = "");
        void close();

        // Function to add single or batched data with a key-value pair
        // Returns true on success, false on failure
        [[nodiscard]] bool put(const size_t& n, const uint64_t* keys, char* values, size_t* value_sizes) const;

        [[nodiscard]] uint64_t* putAutoKey(const size_t& n, char* values, size_t* value_sizes) const;

        // Function to retrieve data by key
        // Returns the value or empty string if the key does not exist
        [[nodiscard]] std::string get(const std::string& key) const;

        // Function to remove data using a key
        // Returns true on success, false if the key does not exist
        [[nodiscard]] bool remove(const uint64_t& key) const;

        [[nodiscard]] bool is_open() const;
    };
}

#endif //MICROVECDB_STORAGE_H
