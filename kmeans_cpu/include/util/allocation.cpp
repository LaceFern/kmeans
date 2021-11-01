
#include "allocation.h"


void * ddr_alloc(size_t bytes){
    return _mm_malloc(bytes, ALIGNMENT);
}

void ddr_free(void *ptrs){
    _mm_free(ptrs);
}
