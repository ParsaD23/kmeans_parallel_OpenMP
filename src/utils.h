#ifndef UTILS_H_
#define UTILS_H_

double** read_dataset(char* filename, int size, int num_features);

double** create_matrix(int rows, int columns);
void free_matrix(double** dataset, int rows, int columns);

double min_matrix(double** array, int rows, int columns);
double max_matrix(double** array, int rows, int columns);

double random_double(double min, double max);
double euclidean_distance(double* a, double* b, int dim);

#endif