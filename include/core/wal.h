#ifndef MICROVECDB_WAL_H
#define MICROVECDB_WAL_H

#include "constants.h"
#include "serializable.h"
#include <mutex>
#include <utility>
#include <fstream>
#include <iostream>

namespace mvdb {

    enum WALEntryType {
        ADD_DATA = 0,
        REMOVE_DATA = 1,
        UPDATE_DATA = 2
    };

    struct WALEntry final : Serializable {
        WALEntryType entry_type = ADD_DATA; // log type
        idx_t nv;   // number of vectors being added/removed
        idx_t dims;
        std::vector<idx_t> ids; // ids to be removed
        std::vector<value_t> values;  // vectors to be added

        void serialize(std::ostream &out) const override {
            serialize_numeric<int8_t>(out, static_cast<int8_t>(entry_type));
            serialize_numeric<idx_t>(out, nv);
            out.write(reinterpret_cast<const char*>(ids.data()), nv * dims * sizeof(idx_t));
            out.write(reinterpret_cast<const char*>(values.data()), nv * dims * sizeof(value_t));
        }

        void deserialize(std::istream &in) override {
            nv = deserialize_numeric<idx_t>(in);
            dims = deserialize_numeric<idx_t>(in);
            in.read(reinterpret_cast<char*>(ids.data()), nv * dims * sizeof(idx_t));
            in.read(reinterpret_cast<char*>(values.data()), nv * dims * sizeof(value_t ));
        }

        WALEntry(idx_t nv, idx_t dims, std::vector<idx_t> ids, std::vector<value_t> values, WALEntryType entry_type) :
                nv(nv), dims(dims), ids(std::move(ids)), values(std::move(values)), entry_type(entry_type) {}
        ~WALEntry() override = default;
    };

    class WAL final : Serializable {
        std::string wal_path;
        std::streampos checkpoint = 0; // Position in the file up to which entries have been processed
        std::mutex wal_mutex; // Ensure thread-safe access
        std::streampos max_size = 1024 * 1024; // default max WAL size = 1MB
    public:
        explicit WAL(std::string  path) : wal_path(std::move(path)) {}

        // Write an entry to the WAL
        void write_entry(const std::string& entry) {
            std::lock_guard<std::mutex> lock(wal_mutex);
            std::ofstream wal_file(wal_path, std::ios::app | std::ios::binary);
            if (wal_file.is_open()) {
                wal_file << entry << std::endl;
                truncate(); // truncate old WAL entries in WAL file has gotten too big
            } else {
                throw std::runtime_error("Failed to open WAL file, '" + wal_path + "' for writing");
            }
        }

        // Mark the current end of the WAL as the checkpoint
        void update_checkpoint() {
            std::lock_guard<std::mutex> lock(wal_mutex);
            std::ifstream walFile(wal_path, std::ios::ate | std::ios::binary);
            if (walFile.is_open()) {
                checkpoint = walFile.tellg();
            } else {
                throw std::runtime_error("Failed to open WAL file, '" + wal_path + "' for reading");
            }
        }

        // Truncate the WAL file up to the checkpoint if it exceeds a certain size
        void truncate() {
            std::ifstream walFile(wal_path, std::ios::ate | std::ios::binary);
            std::streampos fileSize = walFile.tellg();

            if (fileSize > max_size && checkpoint > 0) {
                // Create a new WAL file content that starts from the checkpoint
                walFile.seekg(checkpoint);
                std::string content((std::istreambuf_iterator<char>(walFile)),
                                    std::istreambuf_iterator<char>());

                // Overwrite the WAL file with truncated content
                std::ofstream walFileOut(wal_path, std::ios::trunc | std::ios::binary);
                walFileOut << content;
                // Reset checkpoint since we've truncated the file
                checkpoint = 0;
            }
        }
    };
}

#endif //MICROVECDB_WAL_H
