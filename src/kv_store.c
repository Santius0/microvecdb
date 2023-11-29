#include "kv_store.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>


kv_store_t* kv_store_init(const char* path, const bool create_new, const bool overwrite){

    if (!exists(path)) {               // dir does not exist so must create dir
        if (!create_new) {             // but create_new set to false
            fprintf(stderr, "Error: Database does not exist at path: %s\n", path);
            return NULL;
        }
    } else {                            // dir exists so must either load from dir or overwrite
        if (create_new && !overwrite) { // cannot create_new with existing dir unless overwrite set to true
            fprintf(stderr, "Error: Database already exists at path: %s\n", path);
            return NULL;
        }
    }

    kv_store_t* store = malloc(sizeof(kv_store_t)); // Allocate memory for the kv_store_t instance
    if (!store) {
        fprintf(stderr, "Memory Allocation Error: Failed to allocate (%lu) for kv_store\n", sizeof(kv_store_t));
        return NULL;
    }

    store->db_path = strdup(path);
    store->options = rocksdb_options_create(); // Using default options for now
    store->read_options = rocksdb_readoptions_create();
    store->write_options = rocksdb_writeoptions_create();
    rocksdb_options_set_create_if_missing(store->options, 1);

    if (create_new && overwrite) {
        // Delete the old database if it exists and overwrite is true
        rocksdb_destroy_db(store->options, path, NULL);
    }

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
        if (store->db) rocksdb_close(store->db);
        if (store->options) rocksdb_options_destroy(store->options);
        if (store->write_options) rocksdb_writeoptions_destroy(store->write_options);
        if (store->read_options) rocksdb_readoptions_destroy(store->read_options);
        if (store->db_path) free(store->db_path);
        free(store);
        store = NULL;
    }
}

bool kv_store_put(const kv_store_t* store, kv_pair_t kv_pair, const bool auto_get_key) {
    if(auto_get_key && (!store || !kv_pair.value)) {
        if(!kv_pair.key) kv_pair.key = malloc(sizeof(char));
        kv_pair.key[0] = '\0';
        return false;
    }

    if(!auto_get_key && (!store || !kv_pair.key || !kv_pair.value)) return false;

    if(auto_get_key) {
        const uint64_t sequence_num = rocksdb_get_latest_sequence_number(store->db);
        if(!kv_pair.key) kv_pair.key = malloc(MAX_KEY_SIZE_BYTES); // (2^64 - 1) is a number with 20 digits
        sprintf(kv_pair.key, "%llu", sequence_num);
    }

    const size_t key_len = strlen(kv_pair.key) + 1;
    const size_t value_len = strlen(kv_pair.value) + 1;
    char *err = NULL;

    printf("inserting \"%s\" (%lu) --------> \"%s\" (%lu)\n", kv_pair.key, key_len, kv_pair.value, value_len);
    rocksdb_put(store->db, store->write_options, kv_pair.key, key_len, kv_pair.value, value_len, &err);

    if (err) {
        printf("%s", err);
        rocksdb_free(err);
        return false;
    }
    return true;
}

bool kv_store_put_many(const kv_store_t* store, kv_pair_t* pairs, const size_t num_pairs, const bool auto_get_key) {
    if (!store || !pairs || num_pairs == 0) return false;

    if(auto_get_key) {
        const uint64_t sequence_num = rocksdb_get_latest_sequence_number(store->db);
        for(int i = 0; i < num_pairs; i++) {
            if(!pairs[i].key) pairs[i].key = malloc(MAX_KEY_SIZE_BYTES); // if nothing allocated give it max size otherwise assume enough space has been alloacted
            sprintf(pairs[i].key, "%llu", sequence_num + i);
        }
    }

    char *err = NULL;
    rocksdb_writebatch_t* batch = rocksdb_writebatch_create();

    for (size_t i = 0; i < num_pairs; i++) {
        rocksdb_writebatch_put(batch, pairs[i].key, strlen(pairs[i].key) + 1, pairs[i].value, strlen(pairs[i].value) + 1);
    }

    rocksdb_write(store->db, store->write_options, batch, &err);

    rocksdb_writebatch_destroy(batch);

    if (err) {
        rocksdb_free(err);
        return false;
    }

    return true;
}

char* kv_store_get(const kv_store_t* store, const char* key) {
    if (!store || !key) return NULL;

    const size_t key_len = strlen(key) + 1;
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
