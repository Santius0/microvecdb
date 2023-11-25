#ifndef CONFLUXDB_INDEX_ENGINE_H
#define CONFLUXDB_INDEX_ENGINE_H

#include "annoylib.h"
#include "kissrandom.h"

class ConfluxIndex{
private:
    explicit ConfluxIndex(const int &vec_len, const char *path);
    ~ConfluxIndex();

    static ConfluxIndex *instance;

    Annoy::AnnoyIndex<int, float, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy> *idx;
public:
    ConfluxIndex(const ConfluxIndex&) = delete;
    ConfluxIndex& operator=(const ConfluxIndex&) = delete;

    static ConfluxIndex *getInstance(const int &vec_len, const char *path);
    static void destroyInstance();

    bool add(int i, const float* vec);
    bool get(int i, float* vec);
};

#endif //CONFLUXDB_INDEX_ENGINE_H
