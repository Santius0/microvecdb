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

#include <flat_index.h>
#include <iostream>

int main() {
    // Setup
    std::string indexPath = "test_index.bin";
    if (std::filesystem::exists(indexPath))
        std::filesystem::remove(indexPath);
    uint64_t dimensions = 3; // Example: 3-dimensional vectors

    mvdb::FlatIndex index(indexPath, dimensions);

    // Open or create the index
    index.open();
    if (!index.is_open()) {
        std::cerr << "Failed to open the index." << std::endl;
        return 1;
    }
    std::cout << "Index is open and ready for operations." << std::endl;

    // Add data to the index
    std::vector<mvdb::value_t> data = {
            4.0f, 5.0f, 6.0f, // Vector 1
            1.0f, 2.0f, 3.0f, // Vector 2
            7.0f, 8.0f, 9.0f  // Vector 3
    };
    std::vector<mvdb::idx_t> ids(data.size() / dimensions); // IDs for the added vectors

    if (!index.add(data.size() / dimensions, data.data(), ids.data())) {
        std::cerr << "Failed to add data to the index." << std::endl;
        return 1;
    }
    std::cout << "Data added successfully." << std::endl;
    mvdb::idx_t n = 0;
    mvdb::value_t* curr_data = index.get(n, nullptr);
    for(mvdb::idx_t i = 0; i < index.ntotal(); i++)
        std::cout << curr_data[i] << (i % dimensions == 0 &&  i > 0 ? "\n" : " ");
    std::cout << std::endl;

    // Save the index
    index.save();
    std::cout << "Index saved to " << indexPath << std::endl;

    // Close and reopen the index to simulate reloading
    index.close();
    index.open();
    if (!index.is_open()) {
        std::cerr << "Failed to reload the index." << std::endl;
        return 1;
    }
    std::cout << "Index reloaded successfully." << std::endl;

    // Prepare a query vector
    int k = 5;
    int nq = 1;
    mvdb::value_t query[] = {2.0f, 3.0f, 4.0f}; // Example query
    mvdb::idx_t query_ids[nq * k]; // Assuming we want the closest vector to this query
    mvdb::value_t distances[nq * k]; // Distances from the query to the closest vectors

    // Perform the search
    index.search(nq, query, query_ids, distances, k); // Search for the closest vector to the query

    // Output the search results
    for(int i = 0; i < k; i++)
        std::cout << "Closest vector to query is at index " << query_ids[i] << " with distance " << distances[i] << std::endl;

    return 0;
}

