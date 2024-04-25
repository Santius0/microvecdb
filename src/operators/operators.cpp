#include "operators.h"
#include <fstream>

namespace mvdb {

    template <typename T>
    void insert_(DB_<T>* db, const idx_t& n, const idx_t& d, const T* v, const char* bin,
                 const operators::InsertOperatorDataType& input_data_type,
                 size_t* sizes, const std::string* fp) {
        db->status()->set_timestamp();
        db->status()->set_operation_id(operators::OperatorType::INSERT);
        if(input_data_type == operators::InsertOperatorDataType::VECTOR) {
            if(!db->index()) return;
            auto* index_ids = new idx_t[n];
            uint64_t* storage_ids = db->storage()->putAutoKey(n, const_cast<char *>(bin), sizes);
            if(storage_ids && db->index()->add(n, const_cast<T*>(v), index_ids)) {
                db->status()->set_success(true);
                db->status()->set_message("vectors inserted");
            } else if(storage_ids) {
                for(idx_t i = 0; i < n; i++)
                    bool _ = db->storage()->remove(storage_ids[i]);
                db->status()->set_success(false);
                db->status()->set_message("vectors insert failed");
            }
            delete[] index_ids;
            delete[] storage_ids;
            return;
        }
        else if(input_data_type == operators::InsertOperatorDataType::BINARY) {
            auto * temp = new T[n * d];
            operators::embed_(bin, n, sizes, "insert feature extractor here", const_cast<T*>(v), d);
            db->status()->set_success(v != nullptr);
            db->status()->set_message("embed");
            delete[] temp;
            return;
        }
        else {
            std::vector<char> buffer;
            for(idx_t i = 0; i < n; i++) {
                std::ifstream file(fp[i], std::ios::binary | std::ios::ate);
                if (!file.is_open()) {
                    db->status()->set_success(false);
                    db->status()->set_message("Failed to open file: '" + fp[i] + "'");
                    return;
                }
                file.seekg(0, std::ios::end);
                sizes[i] = file.tellg();
                file.seekg(0, std::ios::beg);

                if (!file.read(buffer.data(), (std::streamsize)sizes[i])) {
                    db->status()->set_success(false);
                    db->status()->set_message("Error reading file: '" + fp[i] + "'");
                    return;
                }
                file.close();
            }
            db->status()->set_success(true);
            db->status()->set_message("file data read");
            return;
        }
    }

    template <typename T>
    void embed_(const char* bytes, const idx_t& n, const size_t* sizes, std::string feature_extractor, T* v, const idx_t& d){
        std::cout << "Feature Extractor = " << feature_extractor << std::endl;
    }
}