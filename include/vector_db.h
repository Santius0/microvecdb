#ifndef VECTOR_DB_H
#define VECTOR_DB_H

#ifdef __cplusplus
extern "C" {
#endif

#include "kv_store.h" // Include the key-value store header
    // Forward declarations for vector index related structures and functions
    // will go here, e.g., vector_index.h

    // Define a structure for the vector database
    typedef struct vector_db_t {
        kv_store_t* kv_store;       // Key-Value store component
        // vector_index_t* vector_index; // Vector index component, assuming a structure for vector indexing

        // Additional fields can be added here as needed
    } vector_db_t;

    // Initialize the vector database
    // This function will allocate and initialize both the key-value store and the vector index
    // It should handle all necessary setup for the database to be functional
    vector_db_t* vector_db_init(const char* path);

    // Free the resources used by the vector database
    // This function should properly release all resources, including those used by the key-value store and vector index
    void vector_db_free(vector_db_t* vdb);

    // Function to add a vector to the database
    // The function will store the vector in the key-value store and update the vector index
    // Returns true on success, false on failure
    bool vector_db_add(vector_db_t* vdb, const char* key, const float* vector, size_t vector_length);

    // Function to search for similar vectors
    // Given a query vector, this function will search the vector index to find the most similar vectors
    // Returns a list of keys of similar vectors or NULL on failure
    char** vector_db_search(vector_db_t* vdb, const float* query_vector, size_t query_vector_length, size_t* result_count);

    // Additional functions for vector database management can be added here
    // These might include updating vectors, removing vectors, handling transactions, etc.

#ifdef __cplusplus
}
#endif

#endif // VECTOR_DB_H
