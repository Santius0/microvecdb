//#include <microvecdb.hpp>
//#include <zmq.hpp>
//#include <string>
//#include <iostream>
//#include <thread>
//#include <db.h>



// device #1: ./microvecdb_main tcp://192.168.1.11:5555
// device #1: ./microvecdb_main tcp://192.168.1.10:5555

//#include "constants.h"
//#include "faiss_flat_index.h"
//int main(int argc, char* argv[]) {
//    auto* db = new mvdb::DB("../demo/deep10M_test_db", "deep10M_test_db", 96);
//    std::cout << "deep10 dims = " << db->index()->dims() << std::endl;
////    float vec[5 * 2] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 33};
////    if(db->add_vector(2, vec))
////        std::cout << "added successfully" << std::endl;
//    size_t n = 5;
//    uint64_t* keys = new uint64_t[n];
//    float* vecs = db->get(n, keys);
//    std::cout << "n = " << n << "\nn x dims = " << n * db->index()->dims() << std::endl;
//    for(int i = 0; i < n * db->index()->dims(); i++){
//        if(i % db->index()->dims() == 0)
//            std::cout << i/db->index()->dims() << ". ";
//        std::cout << vecs[i] << (((i+1) % db->index()->dims()) == 0 ? "\n" : " ");
//    }
//    std::cout << db->index()->ntotal() << std::endl;
//    delete db;
//    delete[] keys;
//    delete[] vecs;
//    std::string server_address = "tcp://localhost:5555";
//    if (argc > 1) server_address = argv[1];
//
//    std::thread server(server_thread);
//    std::thread client(client_thread, server_address);
//
//    client.join();
//    server.detach();

//    mvdb::VectorDB *vdb = new mvdb::VectorDB("./test_db", "test_db");
//    vdb->add_data("An agile fox jumps swiftly over the sleeping dog");
//    vdb->add_data("In the forest, a brown bear climbs over a fallen log");
//    const mvdb::SearchResult sr = vdb->search("The fast brown fox jumps over the lazy hound in the forest", 11, true);
//    std::cout << "Search Results -\n" << sr << std::endl;
//    std::cout << vdb;
//    delete vdb;
//    mvdb::Vector* v = new mvdb::Vector(300, 1, nullptr, nullptr, nullptr);
//    mvdb::DBObject* db = new mvdb::Index();
//    db->ff();
//    delete v;
//    auto* micro_vec_db = new mvdb::VectorDB("./test_mvdb", "test_mvdb");
//    micro_vec_db->create_collection("collection1", 300, "../models/cc.en.300.bin");
//    const mvdb::VectorCollection* collection = micro_vec_db->collection("collection1");
//
//     collection->add_data("An agile fox jumps swiftly over the sleeping dog");
//     collection->add_data("A nimble fox quickly leaps over the resting dog");
//     collection->add_data("An agile fox jumps over the sleeping canine");
//     collection->add_data("The fast brown fox jumps over the lazy hound");
//     collection->add_data("Rapidly jumping over a dog, the brown fox is swift");
//     collection->add_data("Sprinting swiftly, the red fox overcomes the resting dog");
//     collection->add_data("The quick blue fox hops over the lazy dog");
//     collection->add_data("Under a bright moon, a fox jumps over a quiet dog");
//     collection->add_data("In the forest, a brown bear climbs over a fallen log");
//     collection->add_data("Sunshine brightens the quiet forest as the deer prance away");
//     collection->add_data("hello");

//    const mvdb::SearchResult sr = collection->search("The fast brown fox jumps over the lazy hound in the forest", 11, true);
//    std::cout << "Search Results -\n" << sr << std::endl;
//    delete micro_vec_db;
//    return 0;
//}

#include <db.h>
#include <iostream>
#include <future>

// Example task that performs a computation
int compute(int x) {
    for(int i = 0; i < 10000000000; i++){
        x += 1;
    }
    return x * 2; // Just an example computation
}

// Example callback function
void onCompletion(int result) {
    std::cout << "Computation result: " << result << std::endl;
}

// Wrapper function that executes the task and then the callback
template<typename Func, typename Callback>
auto runTaskWithCallback(Func task, Callback callback) {
    return [task, callback]() {
        auto result = task(); // Execute the task
        callback(result); // Execute the callback with the result
    };
}
int main() {
    // Example usage
    auto task = []() { return compute(10); };
    auto future = std::async(std::launch::async,
                             runTaskWithCallback(task, onCompletion));

    // The future here is used to wait for the wrapper (and thus the callback) to complete
    // It's not strictly necessary if you don't need to synchronize on the callback's completion

//    auto* db = new mvdb::DB("./test_db", "test_db", 3);
//    // Add data to the index
//    std::vector<mvdb::value_t> data = {
//            4.0f, 5.0f, 6.0f, // Vector 1
//            1.0f, 2.0f, 3.0f, // Vector 2
//            7.0f, 8.0f, 9.0f,  // Vector 3
//    };
//    std::vector<mvdb::idx_t> ids(data.size() / db->index()->dims()); // IDs for the added vectors
//
//    if (!db->add_vector(data.size()/db->index()->dims(), data.data())) {
//        std::cerr << "Failed to add data to the index." << std::endl;
//        return 1;
//    }
//    std::cout << "Data added successfully." << std::endl;
//    mvdb::idx_t n = 0;
//    mvdb::value_t* curr_data = db->get(n, nullptr);
//    std::cout << "ntotal: " << db->index()->ntotal() << std::endl;
//    for(mvdb::idx_t i = 0; i < db->index()->ntotal(); i++)
//        std::cout << curr_data[i] << (i % db->index()->dims() == 0 &&  i > 0 ? "\n" : " ");
//    std::cout << std::endl;
//    delete db;
    return 0;
}