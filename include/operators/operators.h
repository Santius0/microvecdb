#ifndef MICROVECDB_OPERATORS_H
#define MICROVECDB_OPERATORS_H

#include "db.h"
#include "constants.h"
#include "index.h"
#include <string>


namespace mvdb::operators {

    enum OperatorType : unsigned char {
        INSERT = 0,
        RETRIEVE = 1,
        REMOVE = 2,
        UPDATE = 3,
        EMBED = 4,
        PROJECT = 5,
        IDX_SCAN = 6,
        TABLE_SCAN = 7
    };

    enum InsertOperatorDataType : unsigned char {
        VECTOR = 0,
        BINARY = 1,
        FILE = 2
    };

    /** Inserts records into the database. Allows insertion from raw vectors, raw binary data, or a file path.
       // Parameters:
       // - db: Pointer to the database instance.
       // - data: Pointer to the raw data (vectors or binary).
       // - n: Number of records.
       // - d: Dimension of each vector (not used if data is binary).
       // - data_type: Type of the data ('vector', 'binary', 'file').
       // - fp: Path to the file if data_type is 'file'.
       // - sizes: Array of sizes corresponding to each piece of binary data if data_type is 'BINARY'
    **/
    template <typename T = float>
    void insert_(DB_<T>* db, const idx_t &n, const idx_t &d, const void* v = nullptr, const char* bin = nullptr,
                 const InsertOperatorDataType &input_data_type = VECTOR,
                 size_t *sizes = nullptr, const std::string *fp = nullptr);


        /** Retrieves records from the database based on provided IDs.
        // Parameters:
        // - db: Pointer to the database instance.
        // - ids: Array of record IDs to retrieve.
        // - n: Number of records to retrieve.
        **/
        template <typename T = float>
        void _retrieve(DB_<T>* db, const idx_t *ids, const idx_t &n);


        /** Removes records from the database.
        // Parameters:
        // - db: Pointer to the database instance.
        // - ids: IDs of the records to remove.
        // - n: Number of records to remove.
        **/
        template <typename T = float>
        void _remove(DB_<T>* db, const idx_t &ids, const idx_t &n);


        /** Updates records in the database. Supports raw vectors, raw binary data, or a file path for updates.
        // Parameters:
        // - db: Pointer to the database instance.
        // - ids: IDs of the records to update.
        // - data: Pointer to the new data.
        // - n: Number of records to update.
        // - data_type: Type of the data ('vector', 'binary', 'file').
        // - bytes: Raw binary data, used if data_type is 'binary'.
        // - sizes: Size of the binary data, used if data_type is 'binary'.
        // - fp: Path to the file if data_type is 'file'.
        **/
        template <typename T = float>
        void _update(DB_<T>* db, const idx_t &ids, const void *data, const idx_t &n,
                     const InsertOperatorDataType &input_data_type = VECTOR, const char *bytes = nullptr,
                     const idx_t *sizes = nullptr, const std::string &filePath = "");


        /**
         * Embeds binary data into d-dimensional vectors using a specified feature extractor. If the feature extractor
         * requires a specific dimensionality (d), this function updates d accordingly. The resulting vectors are stored in v.
         *
         * @param bytes Pointer to the binary data input.
         * @param n Number of pieces of binary data.
         * @param sizes Array of sizes corresponding to each piece of binary data.
         * @param feature_extractor Name of the feature extraction algorithm to use.
         * @param v Output array where the resulting d-dimensional vectors are stored.
         * @param d Reference to the dimensionality of the vectors; may be modified by the feature extractor.
         */
        template <typename T = float>
        void embed_(const char *bytes, const idx_t &n, const size_t *sizes, std::string feature_extractor, T *v,
                    const idx_t &d);


//    void _project();

        /**
         * Performs a (c, k)-search using a vector index to find the top-k results within the range c times the closest similarity.
         *
         * @param idx Pointer to the index structure.
         * @param q Pointer to the query vectors.
         * @param n Number of query vectors.
         * @param c Multiplicative factor for the range within which to search for similar vectors.
         * @param k Number of top results to return.
         */
        template <typename T = float>
        void _idx_scan(index::Index<T>* idx, const T *q, const idx_t &n, const uint8_t &c, const uint8_t &k);

        /**
         * Searches the entire database for all results within a specified range.
         *
         * @param db Pointer to the database instance.
         * @param r Range parameter specifying how far from a given point results can be considered relevant.
         */
        template <typename T = float>
        void _range_scan(DB_<T> *db, const double &r);

} // mvdb::opertors

#endif //MICROVECDB_OPERATORS_H
