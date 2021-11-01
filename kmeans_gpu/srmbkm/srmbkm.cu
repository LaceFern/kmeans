#include <vector>
#include <stdint.h>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}
#define LARGE_DOUBLE 99999999999
// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

#include "allocation.h"
#include "dataIo.h"
#include "arguments.h"
#include "mckm.h"
#include "timer.h"
#include "cmdlineparser.h"

#define DSINFO_NUM 1
// #define K_NUM 3
// #define BS_NUM 3
// #define SEED_NUM 3
//#define A_NUM 6
#define K_NUM 1
#define BS_NUM 4
#define SEED_NUM 1
#define A_NUM 1

#define GRIDDIM 1024
#define BLOCKDIM 1024

char *fout_root = "/home/tt/kmeans_gpu/results";

typedef struct dataset_info_str{
    char *filename;
    char *filepath;
    char *filetype;
    int n;
    int d;
}dsinfo;


__device__ float warp_reduce_sum(volatile float *dist_array, int lane_id, int thread_idx){
    //由于一个warp内32个线程是并行执行，条件不满足的等待别的线程执行，所以这是顺序进行了归约
    if(lane_id < 16){
        dist_array[thread_idx]+=dist_array[thread_idx + 16];
    }
    if(lane_id < 8){
        dist_array[thread_idx]+=dist_array[thread_idx + 8];
    }
    if(lane_id < 4){
        dist_array[thread_idx]+=dist_array[thread_idx + 4];
    }
    if(lane_id < 2){
        dist_array[thread_idx]+=dist_array[thread_idx + 2];
    }
    if(lane_id < 1){
        dist_array[thread_idx]+=dist_array[thread_idx + 1];
    }
    return dist_array[thread_idx - lane_id];
}


__device__ float warp_reduce_min_dist(volatile float* dist_array, volatile int* index_array, int lane_id, int thread_idx){
    bool is_update;
    if(lane_id < 16){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 16];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 16] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 16] : dist_array[thread_idx];
    }
    if(lane_id < 8){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 8];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 8] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 8] : dist_array[thread_idx];
    }
    if(lane_id < 4){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 4];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 4] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 4] : dist_array[thread_idx];
    }
    if(lane_id < 2){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 2];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 2] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 2] : dist_array[thread_idx];
    }
    if(lane_id < 1){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 1];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 1] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 1] : dist_array[thread_idx];
    }
    // return index_array[thread_idx - lane_id];
    return dist_array[thread_idx - lane_id];
}

__device__ int warp_reduce_min_index(volatile float* dist_array, volatile int* index_array, int lane_id, int thread_idx){
    bool is_update;
    if(lane_id < 16){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 16];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 16] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 16] : dist_array[thread_idx];
    }
    if(lane_id < 8){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 8];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 8] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 8] : dist_array[thread_idx];
    }
    if(lane_id < 4){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 4];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 4] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 4] : dist_array[thread_idx];
    }
    if(lane_id < 2){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 2];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 2] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 2] : dist_array[thread_idx];
    }
    if(lane_id < 1){
        is_update = dist_array[thread_idx] > dist_array[thread_idx + 1];
        index_array[thread_idx] = is_update ? index_array[thread_idx + 1] : index_array[thread_idx];
        dist_array[thread_idx] = is_update ? dist_array[thread_idx + 1] : dist_array[thread_idx];
    }
    // return index_array[thread_idx - lane_id];
    return index_array[thread_idx - lane_id];
}


__global__ void cal_loss_klt32(
    int n, int d, int k, int iterForcenter,
    float* dmatrix, float* means, 
    double* loss){

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int warp_id = thread_idx >> 5;
    int lane_id = thread_idx - (warp_id<<5);

    float dist_old;
    float dist_new;
    __shared__ float dist_array[1024];
    __shared__ int index_array[1024];
    float coor_1;
    float coor_2;
    float sum_dist = 0;
    float loss_dist = 0;
    int index = 0;
    int batch_size = n;

    dist_old = LARGE_DOUBLE;

    for(int iter = 0; iter < iterForcenter; iter++){
        dist_array[thread_idx] = 0;
        if((warp_id < 32) && warp_id < (k - iter * 32)){
            //这个参数i是干什么的->是为了维度并行
            int i = 0;
            for(i = 0; i < ((d + 31)>>5) - 1; i++){
                coor_1 = dmatrix[1l * block_idx * d + i * 32 + lane_id];
                coor_2 = means[(iter * 32 + warp_id) * d + i * 32 + lane_id];
                dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
            }
            if(lane_id < d - 32 * i){
                // printf("d - 32 * i = %d", d - 32 * i);
                //这里是不是少了个循环->并不
                coor_1 = dmatrix[1l * block_idx * d + i * 32 + lane_id];
                coor_2 = means[(iter * 32 + warp_id) * d + i * 32 + lane_id];
                dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
            }
            sum_dist = warp_reduce_sum(dist_array, lane_id, thread_idx);
            //__syncthreads();
            dist_array[thread_idx] = LARGE_DOUBLE;
            __syncthreads();//跨warp并行
            if(lane_id == 0){
                //把有效的dist紧密放置在dist_array中
                dist_array[warp_id] = sum_dist;
            }
            __syncthreads();
            if(warp_id == 0){
                //把有效的index紧密放置在index_array中(这里thread_idx等于lane_id)
                index_array[thread_idx] = lane_id; 
                dist_new = warp_reduce_min_dist(dist_array, index_array, lane_id, thread_idx);
                if(lane_id == 0 && dist_new < dist_old){
                    dist_old = dist_new;
                }
            }
        }
        // printf("thread_idx = %d end!\n", thread_idx);
    }
    if(warp_id == 0 && lane_id == 0){
        atomicAdd(loss, dist_old);
    }
}


__global__ void cal_loss_kst32(
    int n, int d, int k, int sampleInblock,
    float* dmatrix, float* means, 
    double* loss){

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int warp_id = thread_idx >> 5;
    int lane_id = thread_idx - (warp_id<<5);
    __shared__ float dist_array[1024];
    __shared__ int index_array[1024];
    float coor_1;
    float coor_2;
    float sum_dist = 0;
    float loss_dist = 0;
    int index = 0;
    int batch_size = n;

    // printf("thread_idx = %d start!\n", thread_idx);
    if((warp_id < sampleInblock * k) && (warp_id < (batch_size - block_idx * sampleInblock) * k)){
        dist_array[thread_idx] = 0;
        //这个参数i是干什么的->是为了维度并行
        int i = 0;
        for(i = 0; i < ((d + 31)>>5) - 1; i++){
            coor_1 = dmatrix[1l * block_idx * sampleInblock * d + warp_id / k * d + i * 32 + lane_id];
            coor_2 = means[(warp_id % k) * d + i * 32 + lane_id];
            dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
        }
        if(lane_id < d - 32 * i){
            //这里是不是少了个循环->并不
            coor_1 = dmatrix[1l * block_idx * sampleInblock * d + warp_id / k * d + i * 32 + lane_id];
            coor_2 = means[(warp_id % k) * d + i * 32 + lane_id];
            dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
        }
        sum_dist = warp_reduce_sum(dist_array, lane_id, thread_idx);
        //__syncthreads();
        dist_array[thread_idx] = LARGE_DOUBLE;
        __syncthreads();//跨warp并行
        if(lane_id == 0){
            //把有效的dist紧密放置在dist_array中
            dist_array[warp_id / k * 32 + warp_id % k] = sum_dist;
        }
        __syncthreads();
        if(warp_id < sampleInblock && warp_id < (batch_size - block_idx * sampleInblock)){
            //把有效的index紧密放置在index_array中(这里thread_idx等于lane_id)
            index_array[thread_idx] = lane_id; 
            loss_dist = warp_reduce_min_dist(dist_array, index_array, lane_id, thread_idx);
            if(lane_id == 0){
                atomicAdd(loss, loss_dist);
            }
        }
    }
    // printf("thread_idx = %d end!\n", thread_idx);
}

double cuda_cal_loss(int n, int d, int k, FILE *fp, float *d_dmatrix, float *d_means, int threads){
    cudaStream_t stream_1;
    CHECK(cudaStreamCreate(&stream_1));
    
    double *device_loss;
    CHECK(cudaMalloc(&device_loss, 1 * sizeof(double)));
    CHECK(cudaMemset(device_loss, 0, 1 * sizeof(double)));

    // cudaEvent_t start,stop;
    // CHECK(cudaEventCreate(&start));
    // CHECK(cudaEventCreate(&stop));  
    // CHECK(cudaEventRecord(start, stream_1));

    double host_loss = 0;
    //------------------------------------------
    //一个warp有32个thread，一个block有32个warp
    //thread之间是维度并行，warp之间是center并行和sample并行，block之间是sample并行
    if(k <= 32){
        // printf("k = %d start!\n", k);
        int sampleInblock = 32 / k;
        dim3 cal_dist_block((n - 1) / sampleInblock + 1, 1);
        dim3 cal_dist_thread(1024, 1);
        cal_loss_kst32<<<cal_dist_block, cal_dist_thread, 0, stream_1>>>(
            n, d, k, sampleInblock,
            d_dmatrix, d_means,
            device_loss); 
        // cudaCheckError();
        // printf("k = %d end!\n", k);
    }
    else{
        int blockForcenter = k / 32;
        dim3 cal_dist_block(n, 1);
        dim3 cal_dist_thread(1024, 1);
        cal_loss_klt32<<<cal_dist_block, cal_dist_thread, 0, stream_1>>>(
            n, d, k, blockForcenter,
            d_dmatrix, d_means,
            device_loss); 
        // cudaCheckError();
    }
    //------------------------------------------
    // CHECK(cudaEventRecord(stop, stream_1));
    // CHECK(cudaEventSynchronize(start));
    // CHECK(cudaEventSynchronize(stop));
    cudaMemcpy(&host_loss, device_loss, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    // printf("host_loss = %lf\n", host_loss);
    cudaFree(device_loss);
    // cudaEventDestory(start);
    // cudaEventDestory(stop);
    cudaStreamDestroy(stream_1);

    return host_loss;
}

__global__ void cal_dist_kst32(
    float* dmatrix, float* means, 
    double* sum, double* sumE,
    double* len, double* lenE, 
    int k, int d, int batch_size, int batch_id, int sampleInblock){

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int warp_id = thread_idx >> 5;
    int lane_id = thread_idx - (warp_id<<5);
    __shared__ float dist_array[1024];
    __shared__ int index_array[1024];
    float coor_1;
    float coor_2;
    float sum_dist = 0;
    int index = 0;
    if((warp_id < sampleInblock * k) && (warp_id < (batch_size - block_idx * sampleInblock) * k)){
        dist_array[thread_idx] = 0;
        int i;
        for(i = 0; i < ((d + 31)>>5) - 1; i++){
            coor_1 = dmatrix[1l * batch_id * batch_size * d + block_idx * sampleInblock * d + warp_id / k * d + i * 32 + lane_id];
            coor_2 = means[(warp_id % k) * d + i * 32 + lane_id];
            dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
        }
        if(lane_id < d - 32 * i){
            coor_1 = dmatrix[1l * batch_id * batch_size * d + block_idx * sampleInblock * d + warp_id / k * d + i * 32 + lane_id];
            coor_2 = means[(warp_id % k) * d + i * 32 + lane_id];
            dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
        }
        sum_dist += warp_reduce_sum(dist_array, lane_id, thread_idx);
        //__syncthreads();
        dist_array[thread_idx] = LARGE_DOUBLE;
        __syncthreads();
        if(lane_id == 0){
            dist_array[warp_id / k * 32 + warp_id % k] = sum_dist;
        }
        __syncthreads();
        if(warp_id < sampleInblock && warp_id < (batch_size - block_idx * sampleInblock)){
            index_array[thread_idx] = lane_id; 
            index = warp_reduce_min_index(dist_array, index_array, lane_id, thread_idx);
            for(int j = 0; j < (d / 32 + 1); j++){
                if(lane_id < d - 32 * j){
                    float temp = dmatrix[1l * batch_id * batch_size * d + block_idx * sampleInblock * d + warp_id * d + j * 32 + lane_id];
                    atomicAdd(sum + index * d + j * 32 + lane_id, temp);
                    atomicAdd(sumE + index * d + j * 32 + lane_id, temp);
                }
            }
            if(lane_id == 0){
                atomicAdd(len + index, 1);
                atomicAdd(lenE + index, 1);
            }
        }
    }
}

__global__ void cal_dist_klt32(
    float* dmatrix, float* means, 
    double* sum, double* sumE,
    double* len, double* lenE, 
    int k, int d, int batch_size, int batch_id, int iterForcenter){

    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int warp_id = thread_idx >> 5;
    int lane_id = thread_idx - (warp_id<<5);

    float dist_old;
    float dist_new;
    int index_old;
    int index_new;
    __shared__ float dist_array[1024];
    __shared__ int index_array[1024];
    float coor_1;
    float coor_2;
    float sum_dist = 0;

    dist_old = LARGE_DOUBLE;
    index_old = 0;

    for(int iter = 0; iter < iterForcenter; iter++){
        dist_array[thread_idx] = 0;
        if((warp_id < 32) && warp_id < (k - iter * 32)){
            //这个参数i是干什么的->是为了维度并行
            int i = 0;
            for(i = 0; i < ((d + 31)>>5) - 1; i++){
                coor_1 = dmatrix[1l * batch_id * batch_size * d + block_idx * d + i * 32 + lane_id];
                coor_2 = means[(iter * 32 + warp_id) * d + i * 32 + lane_id];
                dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
            }
            if(lane_id < d - 32 * i){
                // printf("d - 32 * i = %d", d - 32 * i);
                //这里是不是少了个循环->并不
                coor_1 = dmatrix[1l * batch_id * batch_size * d + block_idx * d + i * 32 + lane_id];
                coor_2 = means[(iter * 32 + warp_id) * d + i * 32 + lane_id];
                dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
            }
            sum_dist = warp_reduce_sum(dist_array, lane_id, thread_idx);
            //__syncthreads();
            dist_array[thread_idx] = LARGE_DOUBLE;
            __syncthreads();//跨warp并行
            if(lane_id == 0){
                //把有效的dist紧密放置在dist_array中
                dist_array[warp_id] = sum_dist;
            }
            __syncthreads();
            if(warp_id == 0){
                index_array[thread_idx] = lane_id; 
                index_new = warp_reduce_min_index(dist_array, index_array, lane_id, thread_idx);
                dist_new = dist_array[thread_idx - lane_id];

                if(dist_new < dist_old){
                    dist_old = dist_new;
                    index_old = index_new + iter * 32;
                }
            }
        }
    }

    if(warp_id == 0){
        for(int j = 0; j < (d / 32 + 1); j++){
            if(lane_id < d - 32 * j){
                float temp = dmatrix[1l * batch_id * batch_size * d + block_idx * d + j * 32 + lane_id];
                atomicAdd(sum + index_old * d + j * 32 + lane_id, temp);
                atomicAdd(sumE + index_old * d + j * 32 + lane_id, temp);
            }
        }
        if(lane_id == 0){
            atomicAdd(len + index_old, 1);
            atomicAdd(lenE + index_old, 1);
        }
    }
}


__global__ void cal_means(
    float* means, double* sum, double* len,
    int k, int d){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < k * d){
        int cent_id = idx / d;
        if(len[cent_id] >= 1){
            means[idx] = sum[idx] / len[cent_id];
        }
    }
}

__global__ void update(
    double* sum, double* sumE, double* len, double* lenE, 
    int k, int d, float coefficient){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < k * d){
        int cent_id = idx / d;
        sum[idx] = sumE[idx] * coefficient;
        sumE[idx] = 0;
        if(idx % d == 0){
            len[cent_id] = lenE[cent_id] * coefficient;
            lenE[cent_id] = 0;
        }
    }
}

void cuda_run_srmbkm(
    int n, int d, int k,
    int batch_size, double alpha, int epoch_count,
    float *d_dmatrix, float *d_means, 
    double *d_len, double *d_lenE, 
    double *d_sum, double *d_sumE, 
    double *accu_runtime){

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start,stop;
    float cal_time = 0;
    float temp_time;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));   
    // CHECK(cudaEventRecord(start, 0));
    int batch_num = n / batch_size;

    dim3 cal_means_block_num((k * d - 1)/1024 + 1, 1);
    dim3 cal_means_thread_num(1024, 1);

    cudaEventRecord(start, 0);
    for(int i = 0; i < batch_num; i++){
//--------------------------------------------
        // if(i % batch_num == 0){
        //     cudaEventRecord(start, 0);
        // }
        if(k < 32){
            int sampleInblock = 32 / k;
            dim3 cal_dist_block_num((batch_size - 1)/ sampleInblock + 1, 1);
            dim3 cal_dist_thread_num(1024 , 1);
            cal_dist_kst32<<<cal_dist_block_num, cal_dist_thread_num, 0, stream>>>(
                d_dmatrix, d_means, 
                d_sum, d_sumE, 
                d_len, d_lenE, 
                k, d, batch_size, i, sampleInblock);
            // cudaCheckError();
        }
        else{
            int blockForcenter = k / 32;
            dim3 cal_dist_block_num(batch_size, 1);
            dim3 cal_dist_thread_num(1024, 1);
            cal_dist_klt32<<<cal_dist_block_num, cal_dist_thread_num, 0, stream>>>(
                d_dmatrix, d_means, 
                d_sum, d_sumE, 
                d_len, d_lenE, 
                k, d, batch_size, i, blockForcenter);
            // cudaCheckError();
        }

//--------------------------------------------
        cal_means<<<cal_means_block_num, cal_means_thread_num, 0, stream>>>(d_means, d_sum, d_len, k, d);
    }
    cal_means<<<cal_means_block_num, cal_means_thread_num, 0, stream>>>(d_means, d_sumE, d_lenE, k, d);

    float coefficient = (epoch_count + 1) * alpha;    
    dim3 update_block_num((k * d - 1)/1024 + 1, 1);
    dim3 update_thread_num(1024, 1);
    update<<<update_block_num, update_thread_num, 0, stream>>>(d_sum, d_sumE, d_len, d_lenE, k, d, coefficient);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);
    *accu_runtime += temp_time;
    // cudaMemcpy(host_means, device_means, k * d * sizeof(float), cudaMemcpyDeviceToHost);

    // cudaEventDestory(start);
    // cudaEventDestory(stop);
    cudaStreamDestroy(stream);

}

void start_srmbkm(int n, int d, int k, char *filename, float *dmatrix, float *means, int batchsize, int seed, double alpha, int threads){
    char fout[100];
    sprintf(fout, "%s/srmbkm/%s_k%d_sd%d_bs%d_a%.3lf.txt",
            fout_root, filename, k, seed, batchsize, alpha);
    FILE *fp;
    if((fp=fopen(fout,"wt+")) == NULL){
        printf("Cannot open %s!", fout);
        exit(1);
    }

    float* d_dmatrix;
    float* d_means;
    CHECK(cudaMalloc(&d_dmatrix, 1l * n * d * sizeof(float)));
    CHECK(cudaMalloc(&d_means, k * d * sizeof(float)));
    CHECK(cudaMemcpy(d_dmatrix, dmatrix, 1l * n * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_means, means, k * d * sizeof(float), cudaMemcpyHostToDevice));
          
    double* d_len;
    double* d_sum;
    double* d_lenE;
    double* d_sumE;
    CHECK(cudaMalloc(&d_len, k * sizeof(double)));
    CHECK(cudaMalloc(&d_lenE, k * sizeof(double)));
    CHECK(cudaMalloc(&d_sum, k * d * sizeof(double)));
    CHECK(cudaMalloc(&d_sumE, k * d * sizeof(double)));
    CHECK(cudaMemset(d_len, 0, k * sizeof(double)));
    CHECK(cudaMemset(d_lenE, 0, k * sizeof(double)));
    CHECK(cudaMemset(d_sum, 0, k * d * sizeof(double)));
    CHECK(cudaMemset(d_sumE, 0, k * d * sizeof(double)));

    double *accu_runtime = (double*) ddr_alloc(sizeof (double));
    accu_runtime[0] = 0.0;
    int epoch_num = 200;

    printf("epoch_num: %d; n: %d; d: %d; k: %d; batchsize: %d; seed: %d; alpha: %lf\n", \
        epoch_num, n, d, k, batchsize, seed, alpha); 
    fprintf(fp, "epoch_num: %d; n: %d; d: %d; k: %d; batchsize: %d; seed: %d; alpha: %lf\n", \
        epoch_num, n, d, k, batchsize, seed, alpha);
    double loss = cuda_cal_loss(n, d, k, fp, d_dmatrix, d_means, threads);
    printf("initial loss: %lf\n", loss);
    fprintf(fp, "initial loss: %lf\n", loss);
    for(int i = 0; i < epoch_num; i++){
        cuda_run_srmbkm(
            n, d, k, batchsize, alpha, i, 
            d_dmatrix, d_means,
            d_len, d_lenE,
            d_sum, d_sumE,
            accu_runtime);
        if((i < 9) || (i % 10 == 9)){
            loss = cuda_cal_loss(n, d, k, fp, d_dmatrix, d_means, threads);
            fprintf(fp,"iteration:%d, epoch:%d, data:%d, time:%f, loss: %lf\n", \
            (i + 1) * n / batchsize, i + 1, (i + 1) * n, *accu_runtime, loss);  
            printf("iteration:%d, epoch:%d, data:%d, time:%f, loss: %lf\n",  \
            (i + 1) * n / batchsize, i + 1, (i + 1) * n, *accu_runtime, loss);
        }
    }

    fclose(fp);

    // loss = get_sse(n, k, d, dmatrix, means);
    // printf("loss:%lf\n",loss);
    free(accu_runtime);
    cudaFree(d_means);
    cudaFree(d_dmatrix);
    cudaFree(d_sum);
    cudaFree(d_sumE);
    cudaFree(d_len);
    cudaFree(d_lenE);
}

int main(int argc, char** argv) {
    float *means = NULL;
    float *dmatrix = NULL;
    float *input = NULL;
    
    dsinfo dsinfo_arr[DSINFO_NUM] = {
        // {"sift", "/home/tt/kmeans_gpu/dataset/fvecs/sift_base.fvecs", "fvecs", 1000000, 128}
        // {"gist", "/home/tt/kmeans_gpu/dataset/fvecs/gist_base.fvecs", "fvecs", 1000000, 960},
        // {"poker", "/home/tt/kmeans_gpu/dataset/libsvm/poker.t", "libsvm", 1000000, 10},
        {"mnist8m", "/home/tt/kmeans_gpu/dataset/libsvm/mnist8m", "libsvm", 8100000, 784}
        };

    // int k_arr[K_NUM] = {16, 64, 256};
    // int seed_arr[SEED_NUM] = {1, 10, 100};
    // int batchsize_arr[BS_NUM] = {4096, 16384, 65536};
    // float alpha_arr[A_NUM] = {0, 0.01, 0.1, 1, 10, 100};

    int k_arr[K_NUM] = {128};
    int seed_arr[SEED_NUM] = {1};
    int batchsize_arr[BS_NUM] = {1024, 4096, 16384, 65536};
    float alpha_arr[A_NUM] = {0.01};

    int batchsize_max = 65536;
    // int batchsize_arr[BS_NUM] = {983040, 4096, 16384, 65536};
    int threads = 16;
    int para_d = 32;

    for(int dsinfo_idx = 0; dsinfo_idx < DSINFO_NUM; dsinfo_idx++){
        int n = dsinfo_arr[dsinfo_idx].n;
        int d = (dsinfo_arr[dsinfo_idx].d + para_d - 1) / para_d * para_d;

        input = (float*) ddr_alloc(1l * n * d * sizeof (float));
        dmatrix = (float*) ddr_alloc(1l * n * d * sizeof (float)); 
        memset(input, 0 , 1l * n * d * sizeof (float));            // 清空input数组
        if(strcmp(dsinfo_arr[dsinfo_idx].filetype, "fvecs") == 0){
            fvecs_read(dsinfo_arr[dsinfo_idx].filepath, n, d, input);
        }
        else if(strcmp(dsinfo_arr[dsinfo_idx].filetype, "libsvm") == 0){
            libsvm_read(dsinfo_arr[dsinfo_idx].filepath, n, d, input);
        }
        normalization(d, n, input); 

        for(int seed_idx = 0; seed_idx < SEED_NUM; seed_idx++){
            int seed = seed_arr[seed_idx];
            //printf("checkpoint0: seed=%d", seed);
            shuffle_object(input, d, n, dmatrix, seed);//每次iter循环前都把数据集打乱一下，取前minibatch个sample点

            for(int k_idx = 0; k_idx < K_NUM; k_idx++){
                n = n / batchsize_max * batchsize_max;// 重新设置n为各采样点的最大公倍数
                int k = k_arr[k_idx];
                means = (float*) ddr_alloc(1l * k * d * sizeof (float));  // means = clusters

                for(int bs_idx = 0; bs_idx < BS_NUM; bs_idx++){
                    int batchsize = batchsize_arr[bs_idx];

                    for(int a_idx = 0; a_idx < A_NUM; a_idx++){
                        double alpha = alpha_arr[a_idx];
                        initial_centroids(k, d, n, means, dmatrix);
                        start_srmbkm(n, d, k, dsinfo_arr[dsinfo_idx].filename, dmatrix, means, batchsize, seed, alpha, threads); //打印：只打印前10个
                    }
                }
                free(means);
            }
        }
        free(dmatrix);
        free(input);
    }
}




//--------------------------------------------------------------------------------
