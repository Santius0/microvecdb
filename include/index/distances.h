#ifndef MICROVECDB_DISTANCES_H
#define MICROVECDB_DISTANCES_H

#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <utility>

// Function to calculate L2 distance between two vectors
inline float L2_distance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float distance = 0.0;
    float sum = 0.0;

    #pragma omp parallel for reduction(+:sum) // Parallel for-loop with reduction on 'sum'
    for (size_t i = 0; i < vec1.size(); i++) {
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }

//    distance = std::sqrt(sum); // this metric will only be used to compare the distance between points
                                 // so don't really need to use compute on the sqrt step, just need the overall
                                 // magnitude
    return sum;
}

// Function to find the k nearest neighbors
inline std::vector<uint64_t> knn(const std::vector<float>& query, const std::vector<std::vector<float>>& dataset, int k) {
    // Use a max heap to keep track of the k nearest neighbors found so far
    std::priority_queue<std::pair<float, int>> neighbors;

    for (size_t i = 0; i < dataset.size(); ++i) {
        float distance = L2_distance(query, dataset[i]);
        if (neighbors.size() < k) {
            neighbors.emplace(distance, i);
        } else if (distance < neighbors.top().first) {
            neighbors.pop();
            neighbors.emplace(distance, i);
        }
    }

    // Extract the indices of the k nearest neighbors
    std::vector<uint64_t> nearestIndices;
    while (!neighbors.empty()) {
        nearestIndices.push_back(neighbors.top().second);
        neighbors.pop();
    }
    return nearestIndices;
}

#endif //MICROVECDB_DISTANCES_H
