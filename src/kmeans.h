#ifndef KMEANS_H_
#define KMEANS_H_

double** standard_kmeans(double** dataset, int dataset_size, int num_features, int clusters, int max_iterations, int* assignments, int early_stop);
double** median_kmeans(double** dataset, int dataset_size, int num_features, int clusters, int max_iterations, int* assignments, int early_stop);
double** init_centroids(double** dataset, int dataset_size, int num_features, int clusters);
int find_closest_centroid(double* data_point, double** centroids, int clusters, int num_features);
double distance_sum(double* point, double** points, int size, int dim);

#endif