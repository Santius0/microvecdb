#ifndef CONTSTANTS_H
#define CONTSTANTS_H

#pragma once
#include <cstdint>
#include <vector>

namespace mvdb {
    // VectorDB
    #define META_FILE_EXTENSION ".metadata"
    #define META_FILE_EXTENSION_LEN strlen(META_FILE_EXTENSION)

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

    enum DataFormat {
        RAW_TEXT,

        // Image formats
        JPEG,
        PNG,
        GIF,
        BMP,
        TIFF,

        // Text formats
        TXT,
        PDF,
        DOC,
        HTML,
        RTF,

        // Audio formats
        MP3,
        WAV,
        AAC,
        FLAC,
        OGG,

        // Video formats
        MP4,
        AVI,
        MOV,
        WMV,
        MKV
    };


    using pkey_t = uint64_t; // numerical standard for primary keys
    using vec_count_t = uint16_t;  // numerical standard for specifying number of vectors
    using value_t = int8_t;
    using vector_t = value_t*;
}
#endif //CONTSTANTS_H
