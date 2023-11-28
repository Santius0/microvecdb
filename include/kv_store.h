#ifndef KV_STORE_H
#define KV_STORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <rocksdb/c.h>

#define MAX_KEY_CHARS 21
#define MAX_KEY_SIZE_BYTES sizeof(char) * MAX_KEY_CHARS // (2^64 - 1) is a number with 20 digits

    //Key-value store structure
    typedef struct kv_store_t {
        rocksdb_t *db;
        rocksdb_options_t *options;
        rocksdb_readoptions_t *read_options;
        rocksdb_writeoptions_t *write_options;
        char *db_path;
    } kv_store_t;

    typedef struct kv_pair_t {
        char* key;
        char* value;
    } kv_pair_t;

    // Function to initialize the key-value store
    // The implementation will create a singleton database object.
    // If the database is already created, it will return the existing instance.
    kv_store_t* kv_store_init(const char* path, const bool create_new, const bool overwrite);

    // Function to initialize the key-value store
    void kv_store_free(kv_store_t* store);

    // Function to add data with a key-value pair
    // Returns true on success, false on failure.
    bool kv_store_put(const kv_store_t* store, kv_pair_t kv_pair, const bool auto_get_key);

    // Function to batch add data with a key-value pairs
    // Returns true on success, false on failure.
    bool kv_store_put_many(const kv_store_t* store, kv_pair_t* pairs, size_t num_pairs, const bool auto_get_key);

    // Function to retrieve data by key
    // Returns the value or NULL if the key does not exist.
    char* kv_store_get(const kv_store_t* store, const char* key);

    // Function to remove data using a key
    // Returns true on success, false if the key does not exist.
    bool kv_store_remove(const kv_store_t* store, const char* key);

#ifdef __cplusplus
}
#endif

#endif // KV_STORE_H
