//#include "constants.h"
#include "server.h"
#include <zmq.hpp>
#include <iostream>


namespace mvdb {

    [[noreturn]] void server_thread_start() {
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

    void client_thread_start(const std::string& server_address) {
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
}