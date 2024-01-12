#ifndef MICROVECDB_EXCEPTION_H
#define MICROVECDB_EXCEPTION_H

#include <stdexcept>

namespace mvdb {

    class not_implemented : public std::logic_error {
    public:
        not_implemented() : std::logic_error("Not yet implemented") {}
        explicit not_implemented(const std::string &message) : std::logic_error(message) {}
    };

}
#endif //MICROVECDB_EXCEPTIONS_H