
#ifndef ALLOCATION_H
#define ALLOCATION_H

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif


#define ALIGNMENT 64

void * ddr_alloc(size_t bytes);
void ddr_free(void * ptrs);

#endif
