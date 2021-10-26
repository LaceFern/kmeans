
#ifndef DATA_IO_H
#define DATA_IO_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <vector>
#include <stdint.h>
#include <iostream>
#include <fstream>

#define MAX_LINE_LENGTH 2049

void random_init(float *array, const size_t N, const size_t D);
void read_file_int(int *array, const size_t N, const size_t D, const char *filename, bool isBinary);
void read_file_uint8(u_int8_t *array, const size_t N, const size_t D, const char *filename, bool isBinary);
void read_file(float *array, const size_t N, const size_t D, const char *filename, bool isBinary);
void read_file_double(double *array, const size_t N, const size_t D, const char *filename, bool isBinary);
void save_binary_file(float *array, const size_t N, const int D, char filename[]);
void save_text_file(float *array, const size_t N, const int D, char filename[]);
int fvecs_read (const char *fname, int n, int d, float *a);
int libsvm_read(const char* fname, int n, int d, float* a);
#endif
