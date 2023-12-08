#include "micrvecdb.hpp"

int main() {
    auto* micro_vec_db = new mvdb::MicroVecDB("./test_mvdb", "test_mvdb");
    micro_vec_db->create_collection("collection1", 300, "./models/cc.en.300.bin");
    micro_vec_db->collection("collection1")->add_data("hello");
    delete micro_vec_db;
    return 0;
}
