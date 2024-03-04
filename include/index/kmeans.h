#ifndef MICROVECDB_KMEANS_H
#define MICROVECDB_KMEANS_H

#include <Eigen/Dense>

float calculateDistance(const Eigen::VectorXf& vec, const Eigen::VectorXf& centroid) {
    return (vec - centroid).norm();
//    std::string s;
//    s.length();
}

// Implementation of the k-means clustering algorithm
Eigen::MatrixXf kMeansClustering(const std::vector<Eigen::VectorXf>& data, int num_centroids, int max_iterations = 100) {
    int dims = data.front().rows(); // Dimensionality of the data
    Eigen::MatrixXf centroids(dims, num_centroids);

    // Step 1: Initialization - Randomly select 'num_centroids' data points as the initial centroids
    for (int i = 0; i < num_centroids; ++i) {
        centroids.col(i) = data[std::rand() % data.size()];
    }

    std::vector<int> assignments(data.size());
    bool centroids_changed = true;
    int iterations = 0;

    while (centroids_changed && iterations < max_iterations) {
        centroids_changed = false;

        // Step 2: Assignment step - Assign each data point to the nearest centroid
        for (size_t i = 0; i < data.size(); ++i) {
            float min_distance = std::numeric_limits<float>::max();
            int closest_centroid = 0;
            for (int j = 0; j < num_centroids; ++j) {
                float distance = calculateDistance(data[i], centroids.col(j));
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }
            if (assignments[i] != closest_centroid) {
                assignments[i] = closest_centroid;
                centroids_changed = true;
            }
        }

        // Step 3: Update step - Recalculate centroids as the mean of all data points assigned to each centroid
        Eigen::MatrixXf new_centroids = Eigen::MatrixXf::Zero(dims, num_centroids);
        std::vector<int> counts(num_centroids, 0);
        for (size_t i = 0; i < data.size(); ++i) {
            new_centroids.col(assignments[i]) += data[i];
            counts[assignments[i]]++;
        }

        for (int j = 0; j < num_centroids; ++j) {
            if (counts[j] > 0) {
                new_centroids.col(j) /= counts[j];
            } else { // Handle case where a centroid loses all its points
                new_centroids.col(j) = data[std::rand() % data.size()];
            }
        }

        if ((new_centroids - centroids).norm() > 0.001) {
            centroids_changed = true;
        }
        centroids = new_centroids;
        iterations++;
    }

    return centroids;
}

// Function to find the nearest centroid for a vector segment
int findNearestCentroidIndex(const Eigen::VectorXf& segment, const Eigen::MatrixXf& centroids) {
    float min_distance = std::numeric_limits<float>::max();
    int index = -1;
//    std::cout << centroids << std::endl;
//    std::cout << centroids.cols() << std::endl;
    for (int i = 0; i < centroids.cols(); ++i) {
        float distance = calculateDistance(segment, centroids.col(i));
        if (distance < min_distance) {
            min_distance = distance;
            index = i;
        }
    }
    return index;
}

#endif //MICROVECDB_KMEANS_H
