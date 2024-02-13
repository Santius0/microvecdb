//#ifndef MICROVECDB_WAL_H
//#define MICROVECDB_WAL_H
//
//#include "constants.h"
//#include <string>
//
//#include "spdlog/spdlog.h"
//#include "spdlog/async.h"
//#include "spdlog/sinks/basic_file_sink.h"
//
//namespace mvdb {
//    enum WALActionType {
//        ADD_VECTOR = 0,
//        REMOVE_VECTOR = 1,
//        UPDATE_VECTOR = 2
//    };
//
//    struct WALEntry {
////        std::string timestamp;
////        std::string id;
//        WALActionType type;
//        idx_t num_vecs;
//        idx_t dims;
//        idx_t num_elements;
//        value_t* vecs;
//
//        friend std::ostream& operator<<(std::ostream& os, const WALEntry& obj);
//
//        WALEntry() = default;
//        WALEntry(WALActionType type, idx_t num_vecs, idx_t dims, idx_t num_elements, value_t* vecs);
//    };
//
//    class WAL {
//        std::shared_ptr<spdlog::logger> logger;
//        std::string wal_path;
//        std::thread wal_processor_thread;
//        std::mutex wal_mutex;
//        std::condition_variable wal_condition;
//        std::atomic<bool> processing{false};
//        std::atomic<std::streampos> checkpoint{0}; // Atomic to safely update from another thread
//        const std::streampos max_size = 1048576 * 10; // For example, 10 MB
//    public:
//        WAL() = default;
//        WAL(const std::string& wal_path);
//        ~WAL();
//        void log(const std::string& message);
//        void process_wal();
//        bool need_truncation();
//        void truncate();
//        bool checkpoint_at_eof();
//        void move_checkpoint_by(std::streampos value);
//        void update_checkpoint(std::streampos new_pos);
//    };
//}
//
//#endif //MICROVECDB_WAL_H
