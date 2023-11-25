#include "kv_store.h"
#include <stdlib.h>
#include <stdio.h>
#include <rocksdb/c.h>
#include <string.h>

// Define the key-value store structure
struct kv_store_t {
    rocksdb_t *db;
    rocksdb_options_t *options;
    rocksdb_readoptions_t *read_options;
    rocksdb_writeoptions_t *write_options;
    const char *db_path;
};

kv_store_t* kv_store_init(const char* path) {
    // Allocate memory for the kv_store_t instance
    kv_store_t* store = (kv_store_t*)malloc(sizeof(kv_store_t));
    if (!store) return NULL;

    // Initialize RocksDB
    store->db_path = strdup(path);
    store->options = rocksdb_options_create(); // Using default options for now
    store->read_options = rocksdb_readoptions_create();
    store->write_options = rocksdb_writeoptions_create();
    rocksdb_options_set_create_if_missing(store->options, 1);
    char* err = NULL;
    store->db = rocksdb_open(store->options, store->db_path, &err);
    if(err) {
        rocksdb_free(err);
        kv_store_free(store);
        return NULL;
    }
    return store;
}

void kv_store_free(kv_store_t* store) {
    if (store) {
        if (store->db) {
            rocksdb_close(store->db);
        }
        if (store->options) {
            rocksdb_options_destroy(store->options);
        }
        if (store->write_options) {
            rocksdb_writeoptions_destroy(store->write_options);
        }
        if (store->read_options) {
            rocksdb_readoptions_destroy(store->read_options);
        }
        if (store->db_path) {
            free(store->db_path);
        }
        free(store);
    }
}

bool kv_store_put(const kv_store_t* store, const char* key, const char* value) {
    if (!store || !key || !value) return false;

    const size_t key_len = strlen(key);
    const size_t value_len = strlen(value) + 1;
    char *err = NULL;

    rocksdb_put(store->db, store->write_options, key, key_len, value, value_len, &err);

    if (err) {
        printf("%s", err);
        rocksdb_free(err);
        return false;
    }
    return true;
}

void kv_store_put_auto(const kv_store_t* store, const char* value, char* key) {
    if (!store || !value) {
        key[0] = '\0';
        return;
    }

    const uint64_t sequence_num = rocksdb_get_latest_sequence_number(store->db);
    char *sequence_num_str = malloc(sizeof(char) * 21); // (2^64 - 1) is a number with 20 digits
    sprintf(sequence_num_str, "%llu", sequence_num);

    if(kv_store_put(store, sequence_num_str, value)) strcpy(key, sequence_num_str);
    else key[0] = '\0';

    free(sequence_num_str);
}

char* kv_store_get(const kv_store_t* store, const char* key) {
    if (!store || !key) return NULL;

    const size_t key_len = strlen(key);
    size_t value_len;
    char *err = NULL;
    char *value = rocksdb_get(store->db, store->read_options, key, key_len, &value_len, &err);

    if (err) {
        rocksdb_free(err);
        return NULL;
    }
    return value;
}

bool kv_store_remove(const kv_store_t* store, const char* key) {
    if (!store || !key) return false;

    const size_t key_len = strlen(key);
    char *err = NULL;
    rocksdb_delete(store->db, store->write_options, key, key_len, &err);

    if (err) {
        rocksdb_free(err);
        return false;
    }

    return true;
}
