#include "operators.h"
#include "db.h"
#include <fstream>

namespace mvdb {

    void insert_(DB_* db, const idx_t& n, const idx_t& d, const value_t* v, const char* bin, const operators::InsertOperatorDataType& input_data_type, size_t* sizes, const std::string* fp) {
        db->status()->set_timestamp();
        db->status()->set_operation_id(operators::OperatorType::INSERT);
        if(input_data_type == operators::InsertOperatorDataType::VECTOR) {
            if(!db->index()) return;
            auto* index_ids = new idx_t[n];
            uint64_t* storage_ids = db->storage()->putAutoKey(n, const_cast<char *>(bin), sizes);
            if(storage_ids && db->index()->add(n, const_cast<value_t*>(v), index_ids)) {
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
            auto * temp = new value_t[n * d];
            operators::embed_(bin, n, sizes, "insert feature extractor here", const_cast<value_t *>(v), d);
            db->status()->set_success(true); // need to figure out a way to determine if embed was successful, just assuming it is right now
            db->status()->set_message("embed");
            operators::insert_(db, n, d, v, bin, operators::InsertOperatorDataType::VECTOR, sizes, fp);
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
            operators::insert_(db, n, d, v, buffer.data(), operators::InsertOperatorDataType::BINARY, sizes, fp);
            return;
        }
    }

     void embed_(const char* bytes, const idx_t& n, const size_t* sizes, std::string feature_extractor, value_t* v, const idx_t& d){

    }

}