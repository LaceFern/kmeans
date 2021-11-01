
#ifndef MCKM_H
#define MCKM_H

#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <string.h>

#include <unistd.h> 
#include <stdlib.h>
#include <malloc.h> 
#include <assert.h> 
#include <iostream> 
#include <fstream>      
#include <sstream>     
#include <string> 
#include <limits>

#include "../util/allocation.h"

#define DBL_MAX         1.7976931348623158e+308
#define FLT_MAX 		3.402823466e+38

typedef float vec __attribute__((vector_size(32),aligned(32)));
typedef float vecu __attribute__((vector_size(32)));
typedef unsigned int cidtype;

// void block_km_final_omp_float(const int n, const int k, const int d, const int NUM_THREADS,
//                         float *means, float *dmatrix, const int ITERATIONS,unsigned int* cID);

void compute_reference_kmeans(float* objects, float* clusters_ref,int numObjs, int numClusters, int numCoords, int iter,unsigned int* member_ref, float threshold);
void compare_with_reference(unsigned int* cID, unsigned int* member_ref, float* means, float* means_ref, const int n, const int d, const int k);

float get_change_center_thres (float* features, int nfeatures, int npoints, float thre);
double mini_batch_kmeans_float_4c(const int n, const int k, const int d, const int NUM_THREADS,
                        float *means, float *dmatrix, const int ITERATIONS,unsigned int* cID, float threshold, int* actual_iter, int mini_batch_size);
void shuffle_object (float * objects, int numCoords, int numObjs, float* shuffled_objects,int seed);
float get_sse(int numObjs, int numClusters, int numCoords, float * objects, float * clusters_ref);
void initial_centroids(int numClusters, int numCoords, int numObjs, float* cluster, float* objects);

void normalization(int nfeatures, int npoints, float* features);


void shuffle_object_double (double * objects, int numCoords, int numObjs, double* shuffled_objects,int seed);
double get_sse_double(int numObjs, int numClusters, int numCoords, double * objects, double * clusters_ref);
void initial_centroids_double(int numClusters, int numCoords, int numObjs, double* cluster, double* objects);

void shuffle_object_int (int * objects, int numCoords, int numObjs, int* shuffled_objects,int seed);

void normalization_double(int nfeatures, int npoints, double* features);

void print_label_double(int numObjs, int numClusters, int numCoords, double * objects, double * clusters_ref, const char *filename);
#endif
