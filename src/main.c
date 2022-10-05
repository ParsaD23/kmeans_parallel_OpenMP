#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>
#include <math.h>

#include "kmeans.h"
#include "utils.h"

int main(int argc, char* argv[]) {
	int dataset_size;
	int num_features;
	int clusters;
	int max_iterations;
	char* filename;
	int variant; // 1: kmeans; 2: kmedians
	int early_stop;

	if (argc > 1) {
		filename = argv[1];
	 	dataset_size = atoi(argv[2]);
	 	num_features = atoi(argv[3]);
		clusters = atoi(argv[4]);
		max_iterations = atoi(argv[5]);
		variant = atoi(argv[6]);
		early_stop = (argc == 8 && strcmp(argv[7],"-s") == 0) ? 1 : 0; 
	} else { // Default arguments
		filename = "../data/ex1_4dim_data.csv";
	 	dataset_size = 1500; // correct = 1500; max 3000
	 	num_features = 4; // correct = 4; max 20
	 	clusters = 3; // correct = 3
		max_iterations = 1000;
		early_stop = 0;
	}
	
	double** dataset;

	double** centroids_1;
	int* assignments_1;
	double** centroids_2;
	int* assignments_2;

	double tstart, tstop;

	dataset = read_dataset(filename, dataset_size, num_features);

	assignments_1 = (int*) malloc(dataset_size * sizeof(int));
	assert(assignments_1 != NULL);

	assignments_2 = (int*) malloc(dataset_size * sizeof(int));
	assert(assignments_2 != NULL);
	
	// printf("\n============= KMEANS ============\n");

	if (variant == 1) {	
		tstart = omp_get_wtime();
		centroids_1 = standard_kmeans(dataset, dataset_size, num_features, clusters, max_iterations, assignments_1, early_stop);
		tstop = omp_get_wtime();

		// printf("Kmeans - Elapsed time: %f\n", tstop - tstart);
		printf("%f", tstop - tstart);
		
		free_matrix(centroids_1, clusters, num_features);
		free(assignments_1);
	}

	// printf("\n============ KMEDIANS ===========\n");
	
	if (variant == 2) {
		tstart = omp_get_wtime();
		centroids_2 = median_kmeans(dataset, dataset_size, num_features, clusters, max_iterations, assignments_2, early_stop);
		tstop = omp_get_wtime();

		// printf("Kmedians - Elapsed time: %f\n", tstop - tstart);
		printf("%f", tstop - tstart);
		
		free_matrix(centroids_2, clusters, num_features);
		free(assignments_2);
	}

	free_matrix(dataset, dataset_size, num_features);

	return 0;
}
