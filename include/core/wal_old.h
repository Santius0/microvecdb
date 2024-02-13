//#ifndef MICROVECDB_WAL_OLD_H
//#define MICROVECDB_WAL_OLD_H
//
//#include "constants.h"
//#include "serializable.h"
//#include <mutex>
//#include <utility>
//#include <fstream>
//#include <iostream>
//#include <filesystem>
//#include <atomic>
//#include <condition_variable>
//#include <sstream>
//
//#include "spdlog/spdlog.h"
//#include "spdlog/async.h"
//#include "spdlog/sinks/basic_file_sink.h"
//#include "spdlog/sinks/rotating_file_sink.h"
//
//
//namespace mvdb {
//
//    enum WALEntryType {
//        ADD_DATA = 0,
//        REMOVE_DATA = 1,
//        UPDATE_DATA = 2
//    };
//
//    struct WALEntry {
//        WALEntryType entry_type = ADD_DATA; // log type
//        idx_t nv;   // number of vectors being added/removed
//        idx_t dims;
//        std::vector<idx_t> ids; // ids to be removed
//        std::vector<value_t> values;  // vectors to be added
//
//        [[nodiscard]] std::string serialize() const {
//            std::string data = std::to_string(entry_type) + " " + std::to_string(nv) + " " + std::to_string(dims) + " ";
//            for (auto v : values)
//                data += std::to_string(v) + " ";
//            data += "\n";
//            return data;
//        }
//
//        static WALEntry deserialize(const std::string& data) {
//            WALEntry entry;
//            int type;
//            std::istringstream iss(data);
//            iss >> type;
//            entry.entry_type = static_cast<WALEntryType>(type);
//            iss >> entry.nv;
//            iss >> entry.dims;
//            value_t val;
//            while (iss >> val) {
//                entry.values.push_back(val);
//            }
//        }
//
//        WALEntry() = default;
//        WALEntry(idx_t nv, idx_t dims, std::vector<idx_t> ids, std::vector<value_t> values, WALEntryType entry_type) :
//                nv(nv), dims(dims), ids(std::move(ids)), values(std::move(values)), entry_type(entry_type) {}
//        ~WALEntry() = default;
//    };
//
//    class WALSubscriber {
//    public:
//        WALSubscriber() = default;
//        void notify(const std::string& wal_path);
//    };
//
//    class WAL {
//    private:
//        std::vector<WALSubscriber> subscribers;
//
//        std::shared_ptr<spdlog::logger> logger;
//        std::string log_path;
//        std::mutex log_mutex;
//        std::condition_variable log_condition;
//        std::atomic<bool> stop_logging{false};
//        std::atomic<std::streampos> checkpoint{0}; // Atomic to safely update from another thread
//        const std::streampos max_size = 1048576 * 10; // For example, 10 MB
//
//        // Use a separate thread for log processing
//        std::thread log_thread;
//
//        void logProcessor() {
//            while (!stop_logging) {
//                std::unique_lock<std::mutex> lock(log_mutex);
//                log_condition.wait(lock, [this] { return stop_logging || needsTruncation(); });
//                if (needsTruncation()) {
//                    truncateLog();
//                }
//            }
//        }
//
//        bool needsTruncation() {
//            return(std::filesystem::file_size(log_path) > max_size) && (checkpoint.load() > 0);
//        }
//
//        void truncateLog() {
//            std::lock_guard<std::mutex> lock(log_mutex);
//            std::string temp_log_path = log_path + ".tmp";
//            std::ifstream originalLog(log_path, std::ios::binary);
//            std::ofstream truncatedLog(temp_log_path, std::ios::binary | std::ios::trunc);
//            if (!originalLog.is_open() || !truncatedLog.is_open()) {
//                spdlog::error("Failed to open log files for truncation.");
//                return;
//            }
//            originalLog.seekg(checkpoint.load());
//            truncatedLog << originalLog.rdbuf();
//            originalLog.close();
//            truncatedLog.close();
//            std::filesystem::rename(temp_log_path, log_path);
//            checkpoint = 0;
//            log_condition.notify_all();
//        }
//
//    public:
//        WAL()= default;
//        explicit WAL(std::string  path) : log_path(std::move(path)) {
//            spdlog::init_thread_pool(8192, 1); // Queue size and 1 background thread for logging
//            logger = spdlog::basic_logger_mt<spdlog::async_factory>("wal_logger", log_path);
//            log_thread = std::thread(&WAL::logProcessor, this);
//        }
//
//        ~WAL() {
//            stop_logging = true;
//            log_condition.notify_one();
//            if (log_thread.joinable()) {
//                log_thread.join();
//            }
//        }
//
//        void add_subscriber(WALSubscriber subscriber){
//            subscribers.push_back(subscriber);
//        }
//
//        void notify_subscribers(){
//            for(auto & subscriber : subscribers)
//                subscriber.notify(log_path);
//        }
//
//        void log(const std::string& message) {
//            logger->info(message);
//        }
//
//        void move_checkpoint_by(std::streampos value){
//            update_checkpoint(checkpoint.load() + value);
//        }
//
//        void update_checkpoint(std::streampos new_pos) {
//            checkpoint = new_pos;
//            logger->flush();
//            log_condition.notify_one();
//        }
//    };
//
//}
//
//#endif //MICROVECDB_WAL_OLD_H
