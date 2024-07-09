#ifndef CONTSTANTS_H
#define CONTSTANTS_H

#pragma once
#include <cstdint>
#include <vector>

#define VERSION 0.01

namespace mvdb {

    // VectorDB
    #define DB_EXT "db"
    #define DB_EXT_LEN strlen(DB_EXT)

    // Vector collection
    #define COLLECTION_META_EXT ".meta"
    #define COLLECTION_META_EXT_LEN strlen(COLLECTION_META_EXT)

    // Index store contants
    #define INDEX_EXT "index"
    #define INDEX_EXT_LEN strlen(INDEX_EXT)
    #define INDEX_META_EXT ".meta"
    #define INDEX_META_EXT_LEN strlen(INDEX_META_EXT)

    // Data store contsants
    #define KV_STORE_EXT "data"
    #define KV_STORE_EXT_LEN strlen(KV_STORE_EXT)
    #define MAX_KEY_CHARS 21
    #define MAX_KEY_SIZE_BYTES (sizeof(char) * MAX_KEY_CHARS) // (2^64 - 1) is a number with 20 digits

    using idx_t = int64_t; // all vector indices within an index are counted using a 64-bit unsigned int => can store up to 2^64 vectors per index
}
#endif //CONTSTANTS_H
