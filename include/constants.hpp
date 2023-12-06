#ifndef CONTSTANTS_H
#define CONTSTANTS_H

namespace mvdb {
    // MicroVecDB
    #define META_FILE_NAME "metadata"
    #define META_FILE_NAME_LEN strlen(META_FILE_NAME)

    // Vector collection
    #define COLLECTION_META_EXT ".meta"
    #define COLLECTION_META_EXT_LEN strlen(COLLECTION_META_EXT)

    // Index store contants
    #define INDEX_EXT ".index"
    #define INDEX_EXT_LEN strlen(INDEX_EXT)
    #define INDEX_META_EXT ".index.meta"
    #define INDEX_META_EXT_LEN strlen(INDEX_META_EXT)

    // Data store contsants
    #define KV_STORE_EXT ".data"
    #define KV_STORE_EXT_LEN strlen(KV_STORE_EXT)
    #define MAX_KEY_CHARS 21
    #define MAX_KEY_SIZE_BYTES (sizeof(char) * MAX_KEY_CHARS) // (2^64 - 1) is a number with 20 digits
}
#endif //CONTSTANTS_H
