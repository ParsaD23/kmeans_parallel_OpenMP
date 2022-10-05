#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

double** read_dataset(char* filename, int size, int num_features) {
	FILE* stream = fopen(filename, "r");

	if (stream == NULL) {
		perror("FAILED reading dataset !!!");
		exit(1);
	}

	double** dataset = create_matrix(size, num_features);
	char line[1024];
	
	for (int i = 0; i < size; i++) {		
		if (fgets(line, 1024, stream)) {
			char* token = strtok(line, ",");
			for (int j = 0; j<num_features; j++){
				double value = strtod(token, NULL);
				dataset[i][j] = value;
				token = strtok(NULL, ",");
			}
		}
	}

	fclose(stream);

	return dataset;
}

double random_double(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

double** create_matrix(int rows, int columns) {
	double** matrix = (double**) malloc(rows * sizeof(double*));
	for (int i = 0; i<rows; i++){ 
		matrix[i] = (double*) malloc(columns * sizeof(double));
	}

	return matrix;
}

void free_matrix(double** matrix, int rows, int columns) {
	for (int i = 0; i<rows; i++)
		free(matrix[i]);
	free(matrix);
}

double min_matrix(double** array, int rows, int columns) {
	double min_value;

	#pragma omp parallel for collapse(2) reduction(min:min_value)
	for (int i = 0; i<rows; i++) {
		for (int j = 0; j<columns; j++) {
			if (array[i][j] < min_value)
				min_value = array[i][j];
		}
	}

	return min_value;
}

double max_matrix(double** array, int rows, int columns) {
	double max_value;

	#pragma omp parallel for collapse(2) reduction(max:max_value)
	for (int i = 0; i<rows; i++) {
		for (int j = 0; j<columns; j++) {
			if (array[i][j] > max_value)
				max_value = array[i][j];
		}
	}

	return max_value;
}

double euclidean_distance(double* a, double* b, int dim) {
	double result = 0.0;

	for (int i = 0; i<dim; i++)
		result += (a[i] - b[i])*(a[i] - b[i]);

	return sqrt(result);
}
