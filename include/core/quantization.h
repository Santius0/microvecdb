#ifndef MICROVECDB_QUANTIZATION_H
#define MICROVECDB_QUANTIZATION_H

#include "constants.h"
#include "kmeans.h"
#include <Eigen/Dense>
#include <string>
#include <cmath>

namespace mvdb {
    
    /**
     * @brief Generates centroids from applying product quantization to a dataset.
     *
     * Detailed description of the function, if necessary. This can
     * include the function's overall behavior, any algorithms it uses,
     * and its relevance to other functions or components in the software.
     *
     * @param n Number of vectors passed in as input.
     * @param l Length of each vector.
     * @param v Array containing input vectors. Size of array = n * s.
     * @param s Number of segments each vector should be broken into.
     * @param c Number of k-means centroids for each vector segment.
     * @param vq quantized dataset.
     * @return Description of the return value and what it represents.
     *
     * @note Any important notes about the function, such as assumptions, constraints,
     * or specific behaviors that the caller should be aware of.
     *
     * @example Optional example usage of the function to help developers understand how to use it.
     */
    std::vector<Eigen::MatrixXf> product_quantization_get_centroids(const idx_t& n, const idx_t& l, const std::vector<Eigen::VectorXf>& v, const idx_t& s, const idx_t& c, int8_t vq){
        std::vector<Eigen::MatrixXf> all_centroids(s);
        // Divide vectors into segments and perform k-means clustering for each segment
        for (int i = 0; i < s; ++i) {
            std::vector<Eigen::VectorXf> segment_data;
            const idx_t segment_length = l/s;
            for (const auto& vec : v) {
                Eigen::VectorXf segment = vec.segment(i * segment_length, segment_length);
                segment_data.push_back(segment);
            }
            // Perform k-means clustering on the current segment
            all_centroids[i] = kMeansClustering(segment_data, c);
            // Print out the segment number and its centroids
//            std::cout << "Segment " << i + 1 << " Centroids:" << std::endl;
//            std::cout << all_centroids[i] << std::endl << std::endl;
        }
        return all_centroids;
    }

    void product_quantization_optimize(const idx_t& n, value_t* v, idx_t s, idx_t c, int8_t vq){

    }

    void product_quantization_build_lookup_table(const std::vector<Eigen::MatrixXf>& centroids){
        std::vector<Eigen::VectorXf> combinations;
        Eigen::VectorXf initialCombination(centroids.size() * centroids[0].rows()); // Assuming all segments have the same length
//        generateCombinations(centroids, combinations, initialCombination, 0);

        size_t numCombinations = combinations.size();
        Eigen::MatrixXf distanceMatrix(numCombinations, numCombinations);
    }

}

#endif