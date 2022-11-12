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
#define K_NUM 4
#define BS_NUM 1
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
    //鐢变簬涓€涓獁arp鍐�32涓�绾跨▼鏄�骞惰�屾墽琛岋紝鏉′欢涓嶆弧瓒崇殑绛夊緟鍒�鐨勭嚎绋嬫墽琛岋紝鎵€浠ヨ繖鏄�椤哄簭杩涜�屼簡褰掔害
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
        //杩欎釜鍙傛暟i鏄�骞蹭粈涔堢殑->鏄�涓轰簡缁村害骞惰��
        int i = 0;
        for(i = 0; i < ((d + 31)>>5) - 1; i++){
            coor_1 = dmatrix[1l * block_idx * sampleInblock * d + warp_id / k * d + i * 32 + lane_id];
            coor_2 = means[(warp_id % k) * d + i * 32 + lane_id];
            dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
        }
        if(lane_id < d - 32 * i){
            //杩欓噷鏄�涓嶆槸灏戜簡涓�寰�鐜�->骞朵笉
            coor_1 = dmatrix[1l * block_idx * sampleInblock * d + warp_id / k * d + i * 32 + lane_id];
            coor_2 = means[(warp_id % k) * d + i * 32 + lane_id];
            dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
        }
        sum_dist = warp_reduce_sum(dist_array, lane_id, thread_idx);
        //__syncthreads();
        dist_array[thread_idx] = LARGE_DOUBLE;
        __syncthreads();//璺╳arp骞惰��
        if(lane_id == 0){
            //鎶婃湁鏁堢殑dist绱у瘑鏀剧疆鍦╠ist_array涓�
            dist_array[warp_id / k * 32 + warp_id % k] = sum_dist;
        }
        __syncthreads();
        if(warp_id < sampleInblock && warp_id < (batch_size - block_idx * sampleInblock)){
            //鎶婃湁鏁堢殑index绱у瘑鏀剧疆鍦╥ndex_array涓�(杩欓噷thread_idx绛変簬lane_id)
            index_array[thread_idx] = lane_id; 
            loss_dist = warp_reduce_min_dist(dist_array, index_array, lane_id, thread_idx);
            if(lane_id == 0){
                atomicAdd(loss, loss_dist);
            }
        }
    }
    // printf("thread_idx = %d end!\n", thread_idx);
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
            //杩欎釜鍙傛暟i鏄�骞蹭粈涔堢殑->鏄�涓轰簡缁村害骞惰��
            int i = 0;
            for(i = 0; i < ((d + 31)>>5) - 1; i++){
                coor_1 = dmatrix[1l * block_idx * d + i * 32 + lane_id];
                coor_2 = means[(iter * 32 + warp_id) * d + i * 32 + lane_id];
                dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
            }
            if(lane_id < d - 32 * i){
                // printf("d - 32 * i = %d", d - 32 * i);
                //杩欓噷鏄�涓嶆槸灏戜簡涓�寰�鐜�->骞朵笉
                coor_1 = dmatrix[1l * block_idx * d + i * 32 + lane_id];
                coor_2 = means[(iter * 32 + warp_id) * d + i * 32 + lane_id];
                dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
            }
            sum_dist = warp_reduce_sum(dist_array, lane_id, thread_idx);
            //__syncthreads();
            dist_array[thread_idx] = LARGE_DOUBLE;
            __syncthreads();//璺╳arp骞惰��
            if(lane_id == 0){
                //鎶婃湁鏁堢殑dist绱у瘑鏀剧疆鍦╠ist_array涓�
                dist_array[warp_id] = sum_dist;
            }
            __syncthreads();
            if(warp_id == 0){
                //鎶婃湁鏁堢殑index绱у瘑鏀剧疆鍦╥ndex_array涓�(杩欓噷thread_idx绛変簬lane_id)
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

__global__ void cal_dist_kst32(
    float* dmatrix, float* means, 
    double* sum,
    double* len, 
    int k, int d, int batch_size, int batch_id, int sampleInblock){

    int thread_idx = threadIdx.x;
    int warp_id = thread_idx >> 5;
    int lane_id = thread_idx - (warp_id<<5);
    int block_idx = blockIdx.x;
    __shared__ float dist_array[1024];
    __shared__ int index_array[1024];
    // __shared__ float data_shared[2048];
    __shared__ float center_shared[32 * 128];
    float coor_1;
    float coor_2;

    // for(int sample_offset = 0; sample_offset < sampleInblock; sample_offset ++){
    //     if(thread_idx < d){
    //         data_shared[thread_idx + sample_offset * d] = dmatrix[1l * batch_id * batch_size * d + 1l * block_idx * sampleInblock * d + thread_idx + sample_offset * d];
    //     }
    // }
    // __syncthreads();
    for(int center_offset = 0; thread_idx + center_offset < k * d; center_offset += 1024){
            center_shared[thread_idx + center_offset] = means[thread_idx  + center_offset];
    }
    __syncthreads();
    for( ; (block_idx * sampleInblock) < batch_size; block_idx += gridDim.x){
        float sum_dist = 0;
        int index = 0;
        if((warp_id < sampleInblock * k) && (warp_id < (batch_size - block_idx * sampleInblock) * k)){
            dist_array[thread_idx] = 0;
            int i;
            // int data_offset = warp_id / k * d + lane_id;
            int data_offset = 1l * batch_id * batch_size * d + 1l * block_idx * sampleInblock * d + lane_id +  warp_id / k * d;
            int center_offset = (warp_id % k) * d + lane_id;
            for(i = 0; i < ((d + 31)>>5) - 1; i++){
                coor_1 = dmatrix[data_offset + i * 32];
                // coor_1 = data_shared[data_offset + i * 32];
                // coor_2 = means[center_offset + i * 32];
                coor_2 = center_shared[center_offset + i * 32];
                dist_array[thread_idx] += (coor_1 - coor_2) * (coor_1 - coor_2);
            }
            if(lane_id < d - 32 * i){
                coor_1 = dmatrix[data_offset + i * 32];
                // coor_1 = data_shared[data_offset + i * 32];
                // coor_2 = means[center_offset + i * 32];
                coor_2 = center_shared[center_offset + i * 32];
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
                int data_offset_update = 1l * batch_id * batch_size * d + 1l * block_idx * sampleInblock * d + warp_id * d + lane_id;
                // int data_offset_update = warp_id * d + lane_id;
                int sum_offset_update = index * d + lane_id;
                int j = 0;
                float temp;
                for(j = 0; j <((d + 31)>>5) - 1; j++){
                    temp = dmatrix[data_offset_update + j * 32];
                    // temp = data_shared[data_offset_update + j * 32];
                    atomicAdd(sum + j * 32 + sum_offset_update, temp);
                }
                if(lane_id < d - 32 * j){
                    temp = dmatrix[data_offset_update + j * 32];
                    // temp = data_shared[data_offset_update + j * 32];
                    atomicAdd(sum + j * 32 + sum_offset_update, temp);
                }
                if(lane_id == 0){
                    atomicAdd(len + index, 1);
                }
            }
        }
    }
    
}

__global__ void cal_dist_klt32(
    float* dmatrix, float* means, 
    double* sum, 
    double* len,  
    int k, int d, int batch_size, int batch_id, int iterForcenter){

    int thread_idx = threadIdx.x;
    int warp_id = thread_idx >> 5;
    int lane_id = thread_idx - (warp_id<<5);
    int block_idx =  blockIdx.x;

    float dist_old;
    float dist_new;
    int index_old;
    int index_new;
    __shared__ float dist_array[1024];
    __shared__ int index_array[1024];
    __shared__ float data_shared[1024];
    // __shared__ float center_shared[64 * 128];
    float coor_1;
    float coor_2;
    // float sum_dist = 0;
    
    // for(int center_id = 0; center_id < k; center_id++){
    //     if(thread_idx < d){
    //         center_shared[thread_idx + center_id * d] = means[thread_idx + center_id * d]; 
    //     }
    // }
    // __syncthreads();
    // for( ; block_idx < batch_size; block_idx += gridDim.x){
        float sum_dist = 0;
        int center_iter = 0;

        dist_old = LARGE_DOUBLE;
        index_old = 0;
        float sum_dist_old = LARGE_DOUBLE;

        if(thread_idx < d){
            data_shared[thread_idx] = dmatrix[1l * batch_id * batch_size * d + 1l * block_idx * d + thread_idx];
        }
        __syncthreads();
        
        for(int iter = 0; iter < iterForcenter; iter++){
            dist_array[thread_idx] = 0;
            if(warp_id < (k - iter * 32)){
                //杩欎釜鍙傛暟i鏄�骞蹭粈涔堢殑->鏄�涓轰簡缁村害骞惰��
                int i = 0;
                // int data_offset = 1l * batch_id * batch_size * d + 1l * block_idx * d + lane_id;
                int center_offset = (iter * 32 + warp_id) * d + lane_id;
                float ma_ret = 0;
                for(i = 0; i < ((d + 31)>>5) - 1; i++){
                    // coor_1 = dmatrix[data_offset + i * 32];
                    coor_1 = data_shared[lane_id + i * 32];
                    coor_2 = means[center_offset + i * 32];
                    // coor_2 = center_shared[center_offset + i * 32];
                    ma_ret += (coor_1 - coor_2) * (coor_1 - coor_2);
                }
                if(lane_id < d - 32 * i){
                    // printf("d - 32 * i = %d", d - 32 * i);
                    //杩欓噷鏄�涓嶆槸灏戜簡涓�寰�鐜�->骞朵笉
                    // coor_1 = dmatrix[data_offset + i * 32];
                    coor_1 = data_shared[lane_id + i * 32];
                    coor_2 = means[center_offset + i * 32];
                    // coor_2 = center_shared[center_offset + i * 32];
                    ma_ret += (coor_1 - coor_2) * (coor_1 - coor_2);
                }
                dist_array[thread_idx] = ma_ret;
                sum_dist = warp_reduce_sum(dist_array, lane_id, thread_idx);
                if(sum_dist < sum_dist_old){
                    sum_dist_old = sum_dist;
                    center_iter = iter;
                }
                // __syncthreads();
                // dist_array[thread_idx] = LARGE_DOUBLE;
                // __syncthreads();//璺╳arp骞惰��
                // if(lane_id == 0){
                //     //鎶婃湁鏁堢殑dist绱у瘑鏀剧疆鍦╠ist_array涓�
                //     dist_array[warp_id] = sum_dist;
                // }
                // __syncthreads();
                // if(warp_id == 0){
                //     index_array[thread_idx] = lane_id; 
                //     index_new = warp_reduce_min_index(dist_array, index_array, lane_id, thread_idx);
                //     dist_new = dist_array[thread_idx - lane_id];
    
                //     if(dist_new < dist_old){
                //         dist_old = dist_new;
                //         index_old = index_new + iter * 32;
                //     }
                // }
            }
        }
                // __syncthreads();
                dist_array[thread_idx] = LARGE_DOUBLE;
                __syncthreads();//璺╳arp骞惰��
                if(lane_id == 0){
                    //鎶婃湁鏁堢殑dist绱у瘑鏀剧疆鍦╠ist_array涓�
                    dist_array[warp_id] = sum_dist_old;
                    index_array[warp_id] = warp_id + center_iter * 32; 
                }
                __syncthreads();
    
        if(warp_id == 0){
            index_old = warp_reduce_min_index(dist_array, index_array, lane_id, thread_idx);
    
            int sum_offset_update = index_old * d + lane_id;
            int j = 0;
            float temp;
            for(j = 0; j < ((d + 31)>>5) - 1; j++){
                temp = data_shared[lane_id + j * 32];
    
                atomicAdd(sum + sum_offset_update + j * 32, temp);
            }
            if(lane_id < d - 32 * j){
                // float temp = dmatrix[data_offset_update + j * 32];
                temp = data_shared[lane_id + j * 32];
    
                atomicAdd(sum + sum_offset_update + j * 32, temp);
            }
            if(lane_id == 0){
                atomicAdd(len + index_old, 1);
            }
        }
    //     __syncthreads();
    // }
    
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

//--------------------------------------------------------------------------------

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
    //涓€涓獁arp鏈�32涓猼hread锛屼竴涓猙lock鏈�32涓獁arp
    //thread涔嬮棿鏄�缁村害骞惰�岋紝warp涔嬮棿鏄痗enter骞惰�屽拰sample骞惰�岋紝block涔嬮棿鏄痵ample骞惰��
    if(k <= 32){
        // printf("k = %d start!\n", k);
        int sampleInblock = 32 / k;
        dim3 cal_dist_block((n - 1)/ sampleInblock + 1, 1);
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

void cuda_run_km(
    int n, int d, int k,
    int epoch_count,
    float *d_dmatrix, float *d_means, 
    double* d_len, double* d_sum,
    double *accu_runtime){

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start,stop;
    float cal_time = 0;
    float temp_time;

    CHECK(cudaMemset(d_len, 0, k * sizeof(double)));
    CHECK(cudaMemset(d_sum, 0, k * d * sizeof(double)));

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));   
    // CHECK(cudaEventRecord(start, 0));
    int batch_size = n;
    int batch_num = n / batch_size;

    dim3 cal_means_block_num((k * d - 1)/1024 + 1, 1);
    dim3 cal_means_thread_num(1024, 1);

    // CHECK(cudaEventRecord(start, 0));
    cudaEventRecord(start, 0);
    for(int i = 0; i < batch_num; i++){
//--------------------------------------------
        // if(i % batch_num == 0){
        //     cudaEventRecord(start, 0);
        // }
        if(k < 32){
            int sampleInblock = 32 / k;
            dim3 cal_dist_block_num(160, 1);
            dim3 cal_dist_thread_num(1024 , 1);
            cal_dist_kst32<<<cal_dist_block_num, cal_dist_thread_num, 0, stream>>>(
                d_dmatrix, d_means, 
                d_sum,  
                d_len, 
                k, d, batch_size, i, sampleInblock);
            // cudaCheckError();
        }
        else{
            int blockForcenter = k / 32;
            dim3 cal_dist_block_num(batch_size, 1);
            dim3 cal_dist_thread_num(1024, 1);
            cal_dist_klt32<<<cal_dist_block_num, cal_dist_thread_num, 0, stream>>>(
                d_dmatrix, d_means, 
                d_sum, 
                d_len, 
                k, d, batch_size, i, blockForcenter);
            // cudaCheckError();
        }

//--------------------------------------------
        cal_means<<<cal_means_block_num, cal_means_thread_num, 0, stream>>>(d_means, d_sum, d_len, k, d);
    }

    // CHECK(cudaEventRecord(stop, 0));
    cudaEventRecord(stop, 0);
    // CHECK(cudaEventSynchronize(start));
    cudaEventSynchronize(start);
    // CHECK(cudaEventSynchronize(stop));
    cudaEventSynchronize(stop);
    // CHECK(cudaEventElapsedTime(&temp_time, start, stop));
    cudaEventElapsedTime(&temp_time, start, stop);
    *accu_runtime += temp_time;
    // cudaMemcpy(host_means, device_means, k * d * sizeof(float), cudaMemcpyDeviceToHost);

    // cudaEventDestory(start);
    // cudaEventDestory(stop);
    // CHECK(cudaStreamDestroy(stream));
    cudaStreamDestroy(stream);

} 

void start_km(int n, int d, int k, char *filename, float *dmatrix, float *means, int seed){
    char fout[100];
    sprintf(fout, "%s/km/%s_k%d_sd%d.txt",
            fout_root, filename, k, seed);
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
    CHECK(cudaMalloc(&d_len, k * sizeof(double)));
    CHECK(cudaMalloc(&d_sum, k * d * sizeof(double)));
    CHECK(cudaMemset(d_len, 0, k * sizeof(double)));
    CHECK(cudaMemset(d_sum, 0, k * d * sizeof(double)));

    double *accu_runtime = (double*) ddr_alloc(sizeof (double));
    accu_runtime[0] = 0.0;
    int epoch_num = 50;

    printf("epoch_num: %d; n: %d; d: %d; k: %d; seed: %d\n", \
        epoch_num, n, d, k, seed); 
    fprintf(fp, "epoch_num: %d; n: %d; d: %d; k: %d; seed: %d\n", \
        epoch_num, n, d, k, seed);
    int threads = 16;
    int batchsize = n;
    double loss = cuda_cal_loss(n, d, k, fp, d_dmatrix, d_means, threads);
    printf("initial loss: %lf\n", loss);
    fprintf(fp, "initial loss: %lf\n", loss);
    for(int i = 0; i < epoch_num; i++){
        cuda_run_km(
            n, d, k, i, 
            d_dmatrix, d_means,
            d_len, d_sum,
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
    cudaFree(d_dmatrix);   
    cudaFree(d_means); 
    cudaFree(d_len);   
    cudaFree(d_sum); 
}

int main(int argc, char** argv) {
    float *means = NULL;
    float *dmatrix = NULL;
    float *input = NULL;
    
    dsinfo dsinfo_arr[DSINFO_NUM] = {
        {"sift", "/home/tt/kmeans_gpu/dataset/fvecs/sift_base.fvecs", "fvecs", 1000000, 128},
        // {"gist", "/home/tt/kmeans_gpu/dataset/fvecs/gist_base.fvecs", "fvecs", 1000000, 960},
        // {"poker", "/home/tt/kmeans_gpu/dataset/libsvm/poker.t", "libsvm", 1000000, 10},
        // {"mnist8m", "/home/tt/kmeans_gpu/dataset/libsvm/mnist8m", "libsvm", 8100000, 784}
        };

    // int k_arr[K_NUM] = {16, 64, 256};
    // int seed_arr[SEED_NUM] = {1, 10, 100};
    // int batchsize_arr[BS_NUM] = {4096, 16384, 65536};
    // float alpha_arr[A_NUM] = {0, 0.01, 0.1, 1, 10, 100};

    int k_arr[K_NUM] = {32, 32, 32, 32};
    int seed_arr[SEED_NUM] = {1};

    int batchsize_max = 65536;
    // int batchsize_arr[BS_NUM] = {983040, 4096, 16384, 65536};
    int threads = 16;
    int para_d = 32;

    for(int dsinfo_idx = 0; dsinfo_idx < DSINFO_NUM; dsinfo_idx++){
        int n = dsinfo_arr[dsinfo_idx].n;
        int d = (dsinfo_arr[dsinfo_idx].d + para_d - 1) / para_d * para_d;

        input = (float*) ddr_alloc(1l * n * d * sizeof (float));
        dmatrix = (float*) ddr_alloc(1l * n * d * sizeof (float)); 
        memset(input, 0 , 1l * n * d * sizeof (float));            // 娓呯┖input鏁扮粍
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
            shuffle_object(input, d, n, dmatrix, seed);//姣忔��iter寰�鐜�鍓嶉兘鎶婃暟鎹�闆嗘墦涔变竴涓嬶紝鍙栧墠minibatch涓猻ample鐐�

            for(int k_idx = 0; k_idx < K_NUM; k_idx++){
                n = n / batchsize_max * batchsize_max;// 閲嶆柊璁剧疆n涓哄悇閲囨牱鐐圭殑鏈€澶у叕鍊嶆暟
                int k = k_arr[k_idx];
                means = (float*) ddr_alloc(1l * k * d * sizeof (float));  // means = clusters
                initial_centroids(k, d, n, means, dmatrix);
                start_km(n, d, k, dsinfo_arr[dsinfo_idx].filename, dmatrix, means, seed); //鎵撳嵃锛氬彧鎵撳嵃鍓�10涓�
                free(means);
            }
        }
        free(dmatrix);
        free(input);
    }
}
