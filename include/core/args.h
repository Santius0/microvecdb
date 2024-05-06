#ifndef MICROVECDB_ARGS_H
#define MICROVECDB_ARGS_H

namespace mvdb {
    struct NamedArgs {
        virtual ~NamedArgs() = default;
    };

    struct NoNamedArgs final : NamedArgs {
        NoNamedArgs() = default;
        ~NoNamedArgs() override = default;
    };

    const NamedArgs NoArgs = NoNamedArgs();
}

#endif //MICROVECDB_ARGS_H
