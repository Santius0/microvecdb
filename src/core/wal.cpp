//#include "wal.h"
//#include <fstream>
//
//namespace mvdb {
//
//    // WAL Entry
//    std::ostream& operator<<(std::ostream& os, const WALEntry& obj){
//        os << " " << std::to_string(obj.type) << " " << obj.num_vecs << " "
//           << obj.dims << " " << obj.num_elements << " ";
//        for(idx_t i = 0 ; i <= obj.num_elements - 1; i++)
//            os << obj.vecs[i] << ((i == obj.num_elements) ? "\n" : " ");
//        return os;
//    }
//
//    WALEntry::WALEntry(WALActionType type, idx_t num_vecs, idx_t dims, idx_t num_elements, value_t* vecs):
//    type(type), num_vecs(num_vecs), dims(dims), num_elements(num_elements), vecs(vecs) {}
//
//
//    // WAL
//
//    WAL::WAL(const std::string& path): wal_path(path) {
//        spdlog::init_thread_pool(8192, 1); // Queue size and 1 background thread for logging
//        logger = spdlog::basic_logger_mt<spdlog::async_factory>("wal_logger", wal_path);
//        wal_processor_thread = std::thread(&WAL::process_wal, this);
//    }
//
//    WAL::~WAL(){
////        stop_logging = true;
////            log_condition.notify_one();
////            if (log_thread.joinable()) {
////                log_thread.join();
////            }
//    }
//
//    void WAL::log(const std::string& message) {
//        logger->info(message);
//    }
//
//    bool WAL::checkpoint_at_eof() {
//        std::ifstream wal_file(wal_path);
//        std::streampos file_size = wal_file.tellg();
//        return checkpoint.load() == file_size;
//    }
//
//    void WAL::move_checkpoint_by(std::streampos value){
//        update_checkpoint(checkpoint.load() + value);
//    }
//
//    void WAL::update_checkpoint(std::streampos new_pos) {
//        checkpoint = new_pos;
////        logger->flush();
////        wal_condition.notify_one();
//    }
//
//    void WAL::process_wal() {
//        while (processing) {
//            std::unique_lock<std::mutex> lock(wal_mutex);
//                wal_condition.wait(lock, [this] { return processing || need_truncation(); });
//                if (need_truncation())
//                    truncate();
//                if(!checkpoint_at_eof()){
//
//                }
//            }
//    }
//}