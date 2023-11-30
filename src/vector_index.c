#include "vector_index.h"
#include "utils.h"
#include "constants.h"

#include <faiss/c_api/IndexFlat_c.h>
#include <faiss/c_api/IndexIVFFlat_c.h>
// #include <faiss/c_api/Clustering_c.h>
#include <faiss/c_api/index_io_c.h>
#include <stdlib.h>
#include <string.h>

mvdb_vector_index_t* mvdb_vector_index_create(const char* name, const char* dir, const vector_index_types type, const uint64_t dims) {
    mvdb_vector_index_t* vi = malloc(sizeof(mvdb_vector_index_t));
    if (!vi) return NULL;

    vi->name = strdup(name);
    vi->dir = strdup(dir);
    vi->name_len = strlen(name) + 1;
    vi->dir_len = strlen(dir) + 1;
    vi->dims = dims;
    vi->type = type;

    if(!vi->name || !vi->dir || vi->dims == 0 || vi->name_len == 0 || vi->dir_len == 0) return NULL;

    switch (type) {
        case IVF:
            // For simplicity, using a flat quantizer and a fixed nlist value
            const int nlist = 100;
            FaissIndex* quantizer;
            faiss_IndexFlatL2_new_with(&quantizer, dims);
            faiss_IndexIVFFlat_new_with_metric(&vi->faiss_index, quantizer, dims, nlist, METRIC_L2);
            faiss_Index_free(quantizer); // The index copies the quantizer
        default:
            faiss_IndexFlatL2_new_with(&vi->faiss_index, dims);
    }
    return vi;
}

void mvdb_vector_index_free(mvdb_vector_index_t* vi) {
    if (!vi) return;
    if (vi->faiss_index) faiss_Index_free(vi->faiss_index);
    if(vi->name) free(vi->name);
    if(vi->dir) free(vi->dir);
    free(vi);
    vi = NULL;
}

bool mvdb_vector_index_add(const mvdb_vector_index_t* vi, const size_t n, const float* data){
    if (!vi || !vi->faiss_index || !data) return false;
    faiss_Index_add(vi->faiss_index, n, data);
    return true;
}

bool mvdb_vector_index_remove(const mvdb_vector_index_t* vi, size_t n, const FaissIDSelector* ids) {
    if (!vi || !vi->faiss_index || !ids) return false;
    faiss_Index_remove_ids(vi->faiss_index, ids, &n);
    return true;
}

bool mvdb_vector_index_save(const mvdb_vector_index_t* vi) {
    if (!vi || !vi->faiss_index) return false;

    // paths are dir/collection_name.index and dir/collection_name.index.meta
    // the extra 1 allocated is for the "/"
    const size_t path_preamble_len = strlen(vi->dir) + strlen(vi->name);
    char *index_path = malloc(sizeof(char) * (path_preamble_len + INDEX_EXT_LEN + 1));
    if(!index_path) return false;
    char *index_meta_path = malloc(sizeof(char) * (path_preamble_len + INDEX_META_EXT_LEN + 1));
    if(!index_meta_path) return false;

    sprintf(index_path, "%s/%s%s",vi->dir, vi->name, INDEX_EXT);
    sprintf(index_meta_path, "%s/%s%s",vi->dir, vi->name, INDEX_META_EXT);

    printf("Index saving to: %s\n\n", index_path);
    printf("Index Meta saving to: %s\n\n", index_meta_path);
    printf("Index Meta Being Saved to %s:\n\tname:%s\n\tdir:%s\n\tdims:%lu\n\ttype:%d\n\n",
        index_meta_path, vi->name, vi->dir, vi->dims, vi->type);
    fflush(stdout);

    faiss_write_index_fname(vi->faiss_index, index_path);
    const bool result = vi->faiss_index && !mvdb_vector_index_serialize(vi, index_meta_path);

    if(index_path) free(index_path);
    if(index_meta_path) free(index_meta_path);

    return result;
}

mvdb_vector_index_t* mvdb_vector_index_load(const char* name, const char* dir) {
    const size_t path_preamble_len = strlen(dir) + strlen(name);
    char *index_path = malloc(sizeof(char) * (path_preamble_len + INDEX_EXT_LEN + 1));
    if(!index_path) return false;
    char *index_meta_path = malloc(sizeof(char) * (path_preamble_len + INDEX_META_EXT_LEN + 1));
    if(!index_meta_path) return false;
    sprintf(index_path, "%s/%s%s",dir, name, INDEX_EXT);
    sprintf(index_meta_path, "%s/%s%s",dir, name, INDEX_META_EXT);

    mvdb_vector_index_t* vi = malloc(sizeof(mvdb_vector_index_t));
    if(!vi) return NULL;
    if(!mvdb_vector_index_deserialize(vi, index_meta_path)) {
        mvdb_vector_index_free(vi);
        return NULL;
    }

    faiss_read_index_fname(index_path, 0, &vi->faiss_index);
    if (!vi->faiss_index) {
        mvdb_vector_index_free(vi);
        return NULL;
    }

    printf("Index Meta Loaded From %s = \n\tname:%s\n\tdir:%s\n\tdims:%lu\n\ttype:%d\n\n",
        index_meta_path, vi->name, vi->dir, vi->dims, vi->type);
    fflush(stdout);

    if(index_path) free(index_path);
    if(index_meta_path) free(index_meta_path);

    return vi;
}

bool mvdb_vector_index_deserialize(mvdb_vector_index_t* st, const char* fp) {
    char err_msg[300];
    if(!st) st = malloc(sizeof(mvdb_vector_index_t));
    if(!st) {
        sprintf(err_msg, "Error allocating memory to load vector_index \"%s\"", fp);
        perror(err_msg);
        return false;
    }
    FILE *file = fopen(fp, "rb");
    if (file == NULL) {
        sprintf(err_msg, "Error opening file for reading vector_index \"%s\"", fp);
        perror(err_msg);
        return false;
    }
    // fread(st, sizeof(vector_index_t), 1, file); // Read the struct from the file
    fread(&st->name_len, sizeof(size_t), 1, file);
    fread(&st->dir_len, sizeof(size_t), 1, file);

    st->name = malloc(sizeof(char) * st->name_len);
    st->dir = malloc(sizeof(char) * st->dir_len);

    fread(st->name, sizeof(char), st->name_len, file);
    fread(st->dir, sizeof(char), st->dir_len, file);
    fread(&st->dims, sizeof(u_int64_t), 1, file);
    fread(&st->type, sizeof(int), 1, file);
    fclose(file);
    return true;
}

bool mvdb_vector_index_serialize(const mvdb_vector_index_t* st, const char* fp) {
    FILE *file = fopen(fp, "wb");
    if (file == NULL) {
        char err_msg[300];
        sprintf(err_msg, "Error opening file for writing vector_index \"%s\"", fp);
        perror(err_msg);
        return false;
    }
    // fwrite(st, sizeof(vector_index_t), 1, file); // Write the struct to the file
    fwrite(&st->name_len, sizeof(size_t), 1, file);
    fwrite(&st->dir_len, sizeof(size_t), 1, file);
    fwrite(st->name, sizeof(char), st->name_len, file);
    fwrite(st->dir, sizeof(char), st->dir_len, file);
    fwrite(&st->dims, sizeof(uint64_t), 1, file);
    fwrite(&st->type, sizeof(int), 1, file);
    fclose(file);
    return true;
}
