#!/bin/bash

THREADS=8

gcc -fopenmp ./src/main.c ./src/kmeans.c ./src/utils.c -o ./src/main.o -lm

printf "P\tkmeans\t\tkmedians\n"

for p in `seq $THREADS`; do

	for rep in `seq 1`; do
		printf "$p\t"
		OMP_NUM_THREADS=$p ./src/main.o ./data/ex1_4dim_data.csv 1500 4 3 1000 1
		printf "\t"
		OMP_NUM_THREADS=$p ./src/main.o ./data/ex1_4dim_data.csv 1500 4 3 1000 2
		printf "\n"
	done
done