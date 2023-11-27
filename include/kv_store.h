#ifndef KV_STORE_H
#define KV_STORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdlib.h>

    // Forward declaration for the key-value store structure
    typedef struct kv_store_t kv_store_t;

    // Function to initialize the key-value store
    // The implementation will create a singleton database object.
    // If the database is already created, it will return the existing instance.
    kv_store_t* kv_store_init(const char* path, const bool create_new, const bool overwrite);

    // Function to initialize the key-value store
    void kv_store_free(kv_store_t* store);

    // Function to add data with a key-value pair
    // Returns true on success, false on failure.
    bool kv_store_put(const kv_store_t* store, const char* key, const char* value);

    // Function to retrieve data by key
    // Returns the value or NULL if the key does not exist.
    char* kv_store_get(const kv_store_t* store, const char* key);

    // Function to add data with an automatically assigned key
    // The key will be the next in sequence.
    // Returns the generated key or NULL on failure.
    void kv_store_put_auto(const kv_store_t* store, const char* value, char* key);

    // Function to remove data using a key
    // Returns true on success, false if the key does not exist.
    bool kv_store_remove(const kv_store_t* store, const char* key);

    void* serialize_data(const void* data, size_t data_size, size_t* serialized_size);

    void* deserialize_data(const void* serialized_data, size_t serialized_size, size_t* data_size);

#ifdef __cplusplus
}
#endif

#endif // KV_STORE_H
