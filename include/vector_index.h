#ifndef VECTOR_INDEX_H
#define VECTOR_INDEX_H

#ifdef __cplusplus
extern "C" {
#endif

    #include <faiss/c_api/Index_c.h>
    #include <stdbool.h>

    typedef enum vector_index_types {
        FLAT,
        IVF
    } vector_index_types;

    typedef struct vector_index_t {
        FaissIndex *faiss_index;        // actual faiss index
        uint64_t dims;                  // num dimensions of members in this index
        vector_index_types type;        // type of the vector index
        char *name;
        char *dir;
        size_t name_len;
        size_t dir_len;
    } vector_index_t;

    // Function declarations
    vector_index_t* create_vector_index(const char* name, const char* dir, const vector_index_types type, uint64_t dims);
    void free_vector_index(vector_index_t* vi);
    bool vector_index_add(const vector_index_t* vi, const size_t n, const float* data);
    bool vector_index_remove(const vector_index_t* vi, const size_t n, const FaissIDSelector* ids);
    bool vector_index_save(const vector_index_t* vi);
    vector_index_t* vector_index_load(const char* name, const char* dir) ;
    bool vector_index_deserialize(vector_index_t* st, const char* fp);
    bool vector_index_serialize(const vector_index_t* st, const char* fp);

#ifdef __cplusplus
}
#endif

#endif //VECTOR_INDEX_H
