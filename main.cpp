#include "micrvecdb.hpp"

int main() {
    auto* micro_vec_db = new mvdb::MicroVecDB("./test_mvdb", "test_mvdb");
    micro_vec_db->create_collection("new collection 3", 300, "./models/cc.en.300.bin");
    delete micro_vec_db;
    return 0;
}
