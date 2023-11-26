#include "vector_db.h"

typedef struct vector_db_t {
    kv_store_t* kv_store;       // Key-Value store component
    // vector_index_t* vector_index; // Vector index component, assuming a structure for vector indexing

    // Additional fields can be added here as needed
} vector_db_t;