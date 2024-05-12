#include <iostream>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
//#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>
#include <fstream>


#include "filesystem.h"
//#include "flat_index.h"
#include "faiss_flat_index.h"
#include "distances.h"
#include "db.h"

//constexpr size_t DIMENSIONS = 3;
//constexpr size_t VECTORS_COUNT = 5;
//
//using Vector = std::array<double, DIMENSIONS>;
//
//constexpr Vector vectors[VECTORS_COUNT] = {
//        {1, 2, 3},
//        {4, 5, 6},
//        {7, 8, 9},
//        {10, 11, 12},
//        {13, 14, 15}
//};
//
//constexpr double square(double value) {
//    return value * value;
//}
//
//constexpr double euclideanDistance(const Vector& v1, const Vector& v2) {
//    double sum = 0.0;
//    for (size_t i = 0; i < DIMENSIONS; ++i) {
//        sum += square(v1[i] - v2[i]);
//    }
//    return std::sqrt(sum);
//}
//
//constexpr auto computeDistanceTable() {
//    std::array<std::array<double, VECTORS_COUNT>, VECTORS_COUNT> distanceTable{};
//    for (size_t i = 0; i < VECTORS_COUNT; ++i) {
//        for (size_t j = 0; j < VECTORS_COUNT; ++j) {
//            if (i != j) {
//                distanceTable[i][j] = euclideanDistance(vectors[i], vectors[j]);
//            } else {
//                distanceTable[i][j] = 0.0; // Distance to itself is 0.
//            }
//        }
//    }
//    return distanceTable;
//}
//
////constexpr auto distanceTable = computeDistanceTable();
////
////int main() {
////    std::cout << "Distance between Vector 1 and Vector 2: " << distanceTable[0][1] << std::endl;
////    std::cout << "Distance between Vector 3 and Vector 4: " << distanceTable[2][3] << std::endl;
////    std::cout << "Distance between Vector 2 and Vector 5: " << distanceTable[1][4] << std::endl;
////    return 0;
////}
//
//
//// Helper function to calculate the Euclidean distance between a vector and a centroid
//float calculateDistance(const Eigen::VectorXf& vec, const Eigen::VectorXf& centroid) {
//    return (vec - centroid).norm();
//    std::string s;
//    s.length();
//}
//
//// Implementation of the k-means clustering algorithm
//Eigen::MatrixXf kMeansClustering(const std::vector<Eigen::VectorXf>& data, int num_centroids, int max_iterations = 100) {
//    int dims = data.front().rows(); // Dimensionality of the data
//    Eigen::MatrixXf centroids(dims, num_centroids);
//
//    // Step 1: Initialization - Randomly select 'num_centroids' data points as the initial centroids
//    for (int i = 0; i < num_centroids; ++i) {
//        centroids.col(i) = data[std::rand() % data.size()];
//    }
//
//    std::vector<int> assignments(data.size());
//    bool centroids_changed = true;
//    int iterations = 0;
//
//    while (centroids_changed && iterations < max_iterations) {
//        centroids_changed = false;
//
//        // Step 2: Assignment step - Assign each data point to the nearest centroid
//        for (size_t i = 0; i < data.size(); ++i) {
//            float min_distance = std::numeric_limits<float>::max();
//            int closest_centroid = 0;
//            for (int j = 0; j < num_centroids; ++j) {
//                float distance = calculateDistance(data[i], centroids.col(j));
//                if (distance < min_distance) {
//                    min_distance = distance;
//                    closest_centroid = j;
//                }
//            }
//            if (assignments[i] != closest_centroid) {
//                assignments[i] = closest_centroid;
//                centroids_changed = true;
//            }
//        }
//
//        // Step 3: Update step - Recalculate centroids as the mean of all data points assigned to each centroid
//        Eigen::MatrixXf new_centroids = Eigen::MatrixXf::Zero(dims, num_centroids);
//        std::vector<int> counts(num_centroids, 0);
//        for (size_t i = 0; i < data.size(); ++i) {
//            new_centroids.col(assignments[i]) += data[i];
//            counts[assignments[i]]++;
//        }
//
//        for (int j = 0; j < num_centroids; ++j) {
//            if (counts[j] > 0) {
//                new_centroids.col(j) /= counts[j];
//            } else { // Handle case where a centroid loses all its points
//                new_centroids.col(j) = data[std::rand() % data.size()];
//            }
//        }
//
//        if ((new_centroids - centroids).norm() > 0.001) {
//            centroids_changed = true;
//        }
//        centroids = new_centroids;
//        iterations++;
//    }
//
//    return centroids;
//}
//
//// Function to find the nearest centroid for a vector segment
//int findNearestCentroidIndex(const Eigen::VectorXf& segment, const Eigen::MatrixXf& centroids) {
//    float min_distance = std::numeric_limits<float>::max();
//    int index = -1;
////    std::cout << centroids << std::endl;
////    std::cout << centroids.cols() << std::endl;
//    for (int i = 0; i < centroids.cols(); ++i) {
//        float distance = calculateDistance(segment, centroids.col(i));
//        if (distance < min_distance) {
//            min_distance = distance;
//            index = i;
//        }
//    }
//    return index;
//}
//
//// Recursive function to generate all combinations of centroids
//void generateCombinations(const std::vector<Eigen::MatrixXf>& a_centroids, std::vector<Eigen::VectorXf>& combinations, Eigen::VectorXf currentCombination, int segmentIndex) {
//    if (segmentIndex == a_centroids.size()) {
//        combinations.push_back(currentCombination);
//        return;
//    }
//
//    for (int i = 0; i < a_centroids[segmentIndex].cols(); ++i) {
//        Eigen::VectorXf newCombination = currentCombination;
//        newCombination.segment(segmentIndex * a_centroids[segmentIndex].rows(), a_centroids[segmentIndex].rows()) = a_centroids[segmentIndex].col(i);
//        generateCombinations(a_centroids, combinations, newCombination, segmentIndex + 1);
//    }
//}
//
//// Function to calculate the index in the distance matrix for a given quantized vector
//int calculateIndexForQuantizedVector(const std::vector<int>& quantizedVector, int centroidsPerSegment) {
//    int index = 0;
//    for (size_t i = 0; i < quantizedVector.size(); ++i) {
//        index *= centroidsPerSegment;
//        index += quantizedVector[i];
//    }
//    return index;
//}

//int main() {
//    std::cout << "Hello World making thread..." << std::endl;
//    std::thread thread;
//    // Initialize a set of hardcoded 15-dimensional vectors
//    std::vector<Eigen::VectorXf> vectors = {
//            (Eigen::VectorXf(15) << 1.1, 2.2, 3.3, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0).finished(),
//            (Eigen::VectorXf(15) << 2.2, 3.3, 4.4, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0).finished(),
//            // Add more vectors as needed
//    };
//
//    const int num_segments = 5;
//    const int segment_length = 3;
//    const int num_centroids = 6;
//
//    std::vector<Eigen::MatrixXf> all_centroids(num_segments);
//
//    // Divide vectors into segments and perform k-means clustering for each segment
//    for (int i = 0; i < num_segments; ++i) {
//        std::vector<Eigen::VectorXf> segment_data;
//
//        for (const auto& vec : vectors) {
//            Eigen::VectorXf segment = vec.segment(i * segment_length, segment_length);
//            segment_data.push_back(segment);
//        }
//
//        // Perform k-means clustering on the current segment
//        all_centroids[i] = kMeansClustering(segment_data, num_centroids);
//
//        // Print out the segment number and its centroids
//        std::cout << "Segment " << i + 1 << " Centroids:" << std::endl;
//        std::cout << all_centroids[i] << std::endl << std::endl;
//    }
//
//    Eigen::VectorXf newVector = (Eigen::VectorXf(15) << 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5).finished();
//
//    std::vector<int> quantizedIndices(num_segments);
//
//    // Quantize the new vector
//    for (int i = 0; i < num_segments; ++i) {
//        Eigen::VectorXf segment = newVector.segment(i * segment_length, segment_length);
//        int nearestCentroidIndex = findNearestCentroidIndex(segment, all_centroids[i]);
//        quantizedIndices[i] = nearestCentroidIndex;
//
//        std::cout << "Segment " << i + 1 << ", Nearest Centroid Index: " << nearestCentroidIndex << std::endl;
//    }
//    std::cout << "Quantized Vector: ";
//    for (int i = 0; i < num_segments; ++i)
//        std::cout << quantizedIndices[i] << " ";
//
//
//    std::vector<Eigen::VectorXf> combinations;
//    Eigen::VectorXf initialCombination(all_centroids.size() * all_centroids[0].rows()); // Assuming all segments have the same length
//    generateCombinations(all_centroids, combinations, initialCombination, 0);
//
//    size_t numCombinations = combinations.size();
//    Eigen::MatrixXf distanceMatrix(numCombinations, numCombinations);
//
//    // Compute the Euclidean distance between every pair of combinations
//    for (int i = 0; i < numCombinations; ++i) {
//        for (int j = 0; j < numCombinations; ++j) {
//            if (i == j) {
//                distanceMatrix(i, j) = 0;
//            } else {
//                distanceMatrix(i, j) = calculateDistance(combinations[i], combinations[j]);
//            }
//        }
//    }
//
//    // Print the combinations
//    std::cout << "Combinations:" << std::endl;
//    for (int i = 0; i < combinations.size(); ++i) {
//        std::cout << "Combination " << i + 1 << ": [";
//        for (int j = 0; j < combinations[i].size(); ++j) {
//            std::cout << combinations[i](j);
//            if (j < combinations[i].size() - 1) std::cout << ", ";
//        }
//        std::cout << "]" << std::endl;
//    }
//
//    // Print the distance matrix
//    std::cout << "\nDistance Matrix (" << distanceMatrix.rows() << " x " << distanceMatrix.cols() << ")" << std::endl;
////    for (int i = 0; i < distanceMatrix.rows(); ++i) {
////        for (int j = 0; j < distanceMatrix.cols(); ++j) {
////            // Use std::setw for aligned output if the distances vary in length
////            std::cout << std::setw(8) << distanceMatrix(i, j) << " ";
////        }
////        std::cout << std::endl;
////    }
//
//    // Example quantized vectors (represented as indices of chosen centroids for each segment)
//    std::vector<int> quantizedVector1 = {0, 1, 2, 3, 4}; // Example indices for the first vector
//    std::vector<int> quantizedVector2 = {1, 2, 3, 4, 5}; // Example indices for the second vector
//
//    auto start = std::chrono::steady_clock::now();
//
//
//    Eigen::VectorXf query_vec1 = (Eigen::VectorXf(15) << 3, 6, 8, 1, 3, 4, 7, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5).finished();
//    Eigen::VectorXf query_vec2 = (Eigen::VectorXf(15) << 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5).finished();
//    std::vector<int> quantized_query_vec1(num_segments);
//    std::vector<int> quantized_query_vec2(num_segments);
//
//    for (int i = 0; i < num_segments; ++i) {
//        Eigen::VectorXf segment1 = query_vec1.segment(i * segment_length, segment_length);
//        Eigen::VectorXf segment2 = query_vec2.segment(i * segment_length, segment_length);
//        quantized_query_vec1[i] = findNearestCentroidIndex(segment1, all_centroids[i]);;
//        quantized_query_vec2[i] = findNearestCentroidIndex(segment2, all_centroids[i]);;
//    }
//
//    // Calculate the index in the distance matrix for each quantized vector
//    int index1 = calculateIndexForQuantizedVector(quantized_query_vec1, num_centroids);
//    int index2 = calculateIndexForQuantizedVector(quantized_query_vec2, num_centroids);
//
//    // Assuming distanceMatrix is defined and populated elsewhere
//    // Eigen::MatrixXf distanceMatrix;
//
//    // Extract the distance between the two quantized vectors
//    float distance = distanceMatrix(index1, index2);
//
//    std::cout << "Distance between the two quantized vectors: " << index1 << " and " << index2 << " = " << distance << std::endl;
//
//    auto end = std::chrono::steady_clock::now();
//    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    std::cout << "Time taken: " << elapsed.count() << " microseconds" << std::endl;
//
//    return 0;
//}


#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/AsyncFileReader.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/TruthSet.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/VectorIndex.h"

#include "SSDServing/SSDIndex.h"

int main() {
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    auto ret = SPTAG::VectorIndex::LoadIndex("../SPTAG/Release/sift1m_index_dir", vecIndex);
    if (SPTAG::ErrorCode::Success != ret || nullptr == vecIndex) {
        std::cout << "Cannot open index configure file!");
        return -1;
    }
    SPTAG::SPANN::Options* opts = ((SPTAG::SPANN::Index<float>*)vecIndex.get())->GetOptions();
    if(opts == nullptr) std::cout << "FAIL!" << std::endl;
    else std::cout << "SUCCESS!" << std::endl;
    SPTAG::SSDServing::SSDIndex::Search((SPTAG::SPANN::Index<float>*)(vecIndex.get()));
    return 0;
}