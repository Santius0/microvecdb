#ifndef MICROVECDB_SERVER_H
#define MICROVECDB_SERVER_H

namespace mvdb {
    [[noreturn]] void server_thread_start();
    void client_thread_start();
}

#endif //MICROVECDB_SERVER_H
