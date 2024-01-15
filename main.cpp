//#include <microvecdb.hpp>
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <thread>
#include <db.h>

void server_thread() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("inproc://example");

    while (true) {
        zmq::message_t request;
        socket.recv(&request);
        std::string recv_msg(static_cast<char*>(request.data()), request.size());
        std::cout << "Server received: " << recv_msg << std::endl;

        zmq::message_t reply(5);
        memcpy(reply.data(), "World", 5);
        socket.send(reply);
    }
}

void client_thread(const std::string& server_address) {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);
    socket.connect(server_address); // Connect to the server address

    zmq::message_t request(5);
    memcpy(request.data(), "Hello", 5);
    socket.send(request);

    zmq::message_t reply;
    socket.recv(&reply);
    std::string recv_msg(static_cast<char*>(reply.data()), reply.size());
    std::cout << "Client received: " << recv_msg << std::endl;
}

// device #1: ./microvecdb_main tcp://192.168.1.11:5555
// device #1: ./microvecdb_main tcp://192.168.1.10:5555
#include "constants.h"
int main(int argc, char* argv[]) {
    auto* db = new mvdb::DB("../demo/deep10M_test_db", "deep10M_test_db", 96);
    std::cout << "deep10 dims = " << db->index()->dims() << std::endl;
//    float vec[5 * 2] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 33};
//    if(db->add_vector(2, vec))
//        std::cout << "added successfully" << std::endl;
    size_t n = 5;
    uint64_t* keys = nullptr;
    float* vecs = db->get(n, keys);
    std::cout << "n = " << n << "\nn x dims = " << n * db->index()->dims() << std::endl;
    for(int i = 0; i < n * db->index()->dims(); i++){
        if(i % db->index()->dims() == 0)
            std::cout << i/db->index()->dims() << ". ";
        std::cout << vecs[i] << (((i+1) % db->index()->dims()) == 0 ? "\n" : " ");
    }
    delete db;
    delete[] keys;
    delete[] vecs;
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
    return 0;
}
