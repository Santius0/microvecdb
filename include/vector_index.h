#ifndef VECTOR_INDEX_H
#define VECTOR_INDEX_H

#ifdef __cplusplus
extern "C" {
#endif

    #include <faiss/c_api/Index_c.h>
    #include <stdbool.h>

    typedef enum mvdb_vector_index_types {
        FLAT,
        IVF
    } vector_index_types;

    typedef struct mvdb_vector_index_t {
        FaissIndex *faiss_index;        // actual faiss index
        char *name;                     // index's collection name
        char *dir;                      // index's directory
        size_t name_len;                // size of name. stored for use during deserialization
        size_t dir_len;                 // size of name. stored for use during deserialization
        uint64_t dims;                  // num dimensions of members in this index
        vector_index_types type;        // type of the vector index
    } mvdb_vector_index_t;

    // Function declarations
    mvdb_vector_index_t* mvdb_vector_index_create(const char* name, const char* dir, const vector_index_types type, uint64_t dims);
    void mvdb_vector_index_free(mvdb_vector_index_t* vi);
    bool mvdb_vector_index_add(const mvdb_vector_index_t* vi, const size_t n, const float* data);
    bool mvdb_vector_index_remove(const mvdb_vector_index_t* vi, size_t n, const FaissIDSelector* ids);
    bool mvdb_vector_index_save(const mvdb_vector_index_t* vi);
    mvdb_vector_index_t* mvdb_vector_index_load(const char* name, const char* dir) ;
    bool mvdb_vector_index_deserialize(mvdb_vector_index_t* st, const char* fp);
    bool mvdb_vector_index_serialize(const mvdb_vector_index_t* st, const char* fp);

#ifdef __cplusplus
}
#endif

#endif //VECTOR_INDEX_H
