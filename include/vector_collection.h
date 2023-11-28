#ifndef VECTOR_collection_H
#define VECTOR_collection_H

#ifdef __cplusplus
extern "C" {
#endif

#include "kv_store.h"
#include <stddef.h>

    typedef struct vector_collection_t {
        char *name;
        char *path;
        kv_store_t *kv_store;
    } vector_collection_t;

    // Initialize the vector database
    // This function will allocate and initialize both the key-value store and the vector index
    // It should handle all necessary setup for the database to be functional
    vector_collection_t* vector_collection_init(const char* name, const char* path);

    // Free the resources used by the vector database
    // This function should properly release all resources, including those used by the key-value store and vector index
    void vector_collection_free(vector_collection_t* collection);

    // Function to add a vector to the database
    // The function will store the vector in the key-value store and update the vector index
    // Returns true on success, false on failure
    bool vector_collection_add(vector_collection_t* collection, const char* key, const float* vector, size_t vector_length);

    // Function to search for similar vectors
    // Given a query vector, this function will search the vector index to find the most similar vectors
    // Returns a list of keys of similar vectors or NULL on failure
    char** vector_collection_search(vector_collection_t* collection, const float* query_vector, size_t query_vector_length, size_t* result_count);

    // Additional functions for vector database management can be added here
    // These might include updating vectors, removing vectors, handling transactions, etc.

#ifdef __cplusplus
}
#endif

#endif // VECTOR_collection_H
