//#include <microvecdb.hpp>
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <thread>

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

void client_thread() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);
    socket.connect("inproc://example");

    zmq::message_t request(5);
    memcpy(request.data(), "Hello", 5);
    socket.send(request);

    zmq::message_t reply;
    socket.recv(&reply);
    std::string recv_msg(static_cast<char*>(reply.data()), reply.size());
    std::cout << "Client received: " << recv_msg << std::endl;
}

int main() {
    std::thread server(server_thread);
    std::thread client(client_thread);

    client.join();
    server.detach(); // Server thread will run indefinitely

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
