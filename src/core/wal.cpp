#include "wal.h"
#include <fstream>
#include <iostream>

namespace mvdb {

    // WAL Entry
    std::ostream& operator<<(std::ostream& os, const WALEntry& obj){
        os << obj.timestamp << " " << obj.id << " " << std::to_string(obj.type) << " " << obj.num_vecs << " "
           << obj.dims << " " << obj.num_elements << " ";
        for(idx_t i = 0 ; i <= obj.num_elements - 1; i++)
            os << obj.vecs[i] << ((i == obj.num_elements) ? "\n" : " ");
        return os;
    }

    WALEntry::WALEntry(std::string timestamp, long long int id, WALActionType type, idx_t num_vecs, idx_t dims, idx_t num_elements, value_t* vecs):
    timestamp(timestamp), id(id), type(type), num_vecs(num_vecs), dims(dims), num_elements(num_elements), vecs(vecs) {}
}