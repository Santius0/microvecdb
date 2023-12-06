#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H

#include <ostream>

class Serializable {
protected:
    virtual void serialize(std::ostream& out) const = 0;
    virtual void deserialize(std::istream& in) = 0;
public:
    virtual ~Serializable() = default;
};

#endif //SERIALIZABLE_H
