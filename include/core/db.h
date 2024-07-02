#ifndef DB_H
#define DB_H

#include "constants.h"
#include "serializable.h"
#include "storage.h"
#include "index.h"

#include <unordered_map>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>

namespace mvdb {

    struct Record {
        std::string id;
        std::string rdb_key;
        int64_t idx_key;
    };

    class Status {
    public:
        Status() {
            set_timestamp();
        };
        Status(unsigned char operation_id, bool success, std::string  message, bool ok) :
                operation_id_(operation_id), success_(success), message_(std::move(message)), ok_(ok) {
            set_timestamp();
        }

        [[nodiscard]] std::string getFormattedTimestamp() const {
            auto time_t_format = std::chrono::system_clock::to_time_t(timestamp_);
            std::tm tm = *std::localtime(&time_t_format);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
            return oss.str();
        }

        void printStatus() const {
            std::cout << "Operation ID: " << operation_id_ << "\n"
                      << "Success: " << (success_ ? "Yes" : "No") << "\n"
                      << "Message: " << message_ << "\n"
                      << "OK: " << (ok_ ? "Yes" : "No") << "\n"
                      << "Timestamp: " << getFormattedTimestamp() << std::endl;
        }

        void set_operation_id(unsigned char operation_id) {
            operation_id_ = operation_id;
        }

        void set_success(bool success) {
            success_ = success;
        }

        void set_message(std::string message) {
            message_ = std::move(message);
        }

        void set_ok(bool ok) {
            ok_ = ok;
        }

        void set_timestamp(){
            timestamp_ = std::chrono::system_clock::now();
        }

        [[nodiscard]] unsigned char operation_id() const {
            return operation_id_;
        }

        [[nodiscard]] bool success() const {
            return success_;
        }

        [[nodiscard]] std::string message() const {
            return message_;
        }

        [[nodiscard]] bool ok() const {
            return ok_;
        }

        [[nodiscard]] std::chrono::system_clock::time_point timestamp() const {
            return timestamp_;
        }

    private:
        unsigned char operation_id_{};
        bool success_{};
        std::string message_;
        bool ok_{};
        std::chrono::system_clock::time_point timestamp_;
    };

    template <typename T = float>
    class DB_ final : Serializable {
        std::unique_ptr<Status> status_ = std::make_unique<Status>();
        std::string _path, _db_path, _index_path, _storage_path;
        idx_t _dims = 0;
        index::IndexType _index_type;
        std::unique_ptr<Storage> _storage;
        std::unique_ptr<index::Index<T>> _index;
        std::vector<Record> _records;
        friend std::ostream& operator<<(std::ostream& os, const DB_<T>& obj);
        friend std::ostream& operator<<(std::ostream& os, const DB_<T>* obj);
        void _save(const std::string& save_path = "");
        void index_make(const index::IndexType& index_type);
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        DB_() = default;
        ~DB_() override = default;
        Status* status() const;
        bool open(std::string& path);
        bool create(index::IndexType index_type, const uint64_t& dims, std::string& path,
                    const std::string& initial_data_path = "", const T* initial_data = nullptr,
                    const uint64_t &initial_data_size = 0, const NamedArgs* args = nullptr);

        Storage* storage();
        index::Index<T>* index();
//        T* get(idx_t& n, idx_t* keys) const;
    };

    extern template class DB_<int8_t>;
    extern template class DB_<int16_t>;
    extern template class DB_<uint8_t>;
    extern template class DB_<float>;
}

#endif //DB_H
