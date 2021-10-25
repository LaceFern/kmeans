#include <vector>
#include <stdint.h>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
	
#include "../util/allocation.h"
#include "../util/dataIo.h"
#include "../util/arguments.h"
#include "../mckm/mckm.h"
#include "../util/timer.h"

#include "../cmdparser/cmdlineparser.h"

#include <vector>
#include <stdint.h>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
	
#include "../util/allocation.h"
#include "../util/dataIo.h"
#include "../util/arguments.h"
#include "../mckm/mckm.h"
#include "../util/timer.h"
#include "../cmdparser/cmdlineparser.h"

#define DSINFO_NUM 3
#define K_NUM 1
#define BS_NUM 1
#define MBS_NUM 1
#define SEED_NUM 1
#define A_NUM 1

const int para_d = 16;
const int para_d_db = 8;

using namespace sda::utils;

typedef struct dataset_info_str{
    char *filename;
    char *filepath;
    char *filetype;
    int n;
    int d;
}dsinfo;

char *fout_root = "/home/zxy/final/next_results_8";

#define NUM_THREADS 16
pthread_barrier_t barrier;

typedef struct pthread_info_str{
    double *len;
    double *lenE;
    double *sum;
    double *sumE;
    float *dmatrix;
    float *means;
    int *rindex;
    int par;
    int *bindex;
    int batchsize_pt;
    int batchsize;
    int n;
    int k;
    int d;
    double alpha;
    int iteration;
    int batch_num;
    double *loss;
    int *M0;
    int *M1;
    int *M0_pt;
    int *M1subM0_pt;
    float *sse;
    float *p;
    float *l;
    int *a_old;
    int *a;
    float *self_d;
    float *means_old;
    float *sigmaC;
    int iter_index;
    int mbatchsize_pt;
    int mbatchsize;
    int threads;
}ptinfo;

void *par_task_loss(void *info){
    // pthread_detach(pthread_self());
    ptinfo ptinfo_par = *(ptinfo *)info;
    int batchsize_pt = ptinfo_par.batchsize_pt;
    int k = ptinfo_par.k;
    int d = ptinfo_par.d;
    float *dmatrix = ptinfo_par.dmatrix;
    float *means = ptinfo_par.means;
    int par = ptinfo_par.par;
    int iteration = ptinfo_par.iteration;
    double *loss = ptinfo_par.loss;

    for (int h = 0; h < iteration; h++){
        pthread_barrier_wait(&barrier);

        for (int i = 0; i < batchsize_pt; i++) {
            float min_dist = INFINITY;
            int index = 0;
            for (int j = 0; j < k; j++) {

                __m512 sq_para_d = _mm512_set1_ps(0);
                for (int m = 0; m < d / para_d; m++){
                    long long int tmp_p_bias = 1l * par * batchsize_pt * d + 1l * i * d + m * para_d;
                    int tmp_c_bias = j * d + m * para_d;
                    __m512 p_para_d = _mm512_load_ps((dmatrix + tmp_p_bias));
                    __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias)); 
                    __m512 sub_para_d = _mm512_sub_ps(p_para_d, c_para_d);
                    sq_para_d = _mm512_fmadd_ps(sub_para_d, sub_para_d, sq_para_d);
                }
                float dist = _mm512_reduce_add_ps(sq_para_d);

                if (dist < min_dist) { 
                    min_dist = dist;
                    index = j;
                }
            }
            *loss += min_dist;
        }  
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

void *par_task_nmbkm(void *info){
    ptinfo ptinfo_par = *(ptinfo *)info;
    int k = ptinfo_par.k;
    int d = ptinfo_par.d;
    double *len = ptinfo_par.len;
    double *sum = ptinfo_par.sum;
    float *dmatrix = ptinfo_par.dmatrix;
    float *means = ptinfo_par.means;
    int par = ptinfo_par.par;
    int iteration = ptinfo_par.iteration;
    int *M0_p = ptinfo_par.M0;
    int *M1_p = ptinfo_par.M1;
    float *sse = ptinfo_par.sse;
    float *p = ptinfo_par.p;
    float *l = ptinfo_par.l;
    int *a_old = ptinfo_par.a_old;
    int *a = ptinfo_par.a;
    float *self_d = ptinfo_par.self_d;
    float *means_old = ptinfo_par.means_old;
    float *sigmaC = ptinfo_par.sigmaC;
    int threads = ptinfo_par.threads;

    int iter_index = ptinfo_par.iter_index;

    // if(iter_index == 1)  printf("[par %d] M0:%d, M1:%d, M0_pt:%d, M1subM0_pt:%d \n", par, M0, M1, M0_pt, M1subM0_pt);

    for (int h = 0; h < iteration; h++){
        pthread_barrier_wait(&barrier);
        int M0 = *M0_p;
        int M1 = *M1_p;
        int M0_pt = M0 / threads;
        int M1subM0_pt = (M1 - M0) / threads;

        for (int i = 0; i < M0_pt; i++){
            for (int j = 0; j < k; j++){
                l[1l * par * M0_pt * k + 1l * i * k + j] -= p[j];
            }
        }
        // if(iter_index == 1)  printf("[par %d] check point 0.1 \n", par);
        for (int i = 0; i < M0_pt; i++){
            int tmp_index = par * M0_pt + i;
            a_old[tmp_index] = a[tmp_index];
            // if(iter_index == 1)  printf("[par %d] a[%d]:%d \n", par, tmp_index, a[tmp_index]);
            sse[a_old[tmp_index]] -= self_d[tmp_index] * self_d[tmp_index];
            // if(iter_index == 1)  printf("[par %d] check point 0.1.1 \n", par);
            
            for (int j = 0; j < d / para_d_db; j++){
                long long int tmp_p_bias = 1l * tmp_index * d + j * para_d_db;
                int tmp_s_bias = a_old[tmp_index] * d + j * para_d_db;

                __m256 p_para_d_f = _mm256_load_ps((dmatrix + tmp_p_bias)); 
                float *p_para_d_f_p = (float *)&p_para_d_f;    
                __m512d p_para_d = _mm512_set_pd(*(p_para_d_f_p + 7), \
                                                 *(p_para_d_f_p + 6), \
                                                 *(p_para_d_f_p + 5), \
                                                 *(p_para_d_f_p + 4), \
                                                 *(p_para_d_f_p + 3), \
                                                 *(p_para_d_f_p + 2), \
                                                 *(p_para_d_f_p + 1), \
                                                 *(p_para_d_f_p + 0));
                __m512d s_para_d = _mm512_load_pd((sum + tmp_s_bias));     
                __m512d new_s_para_d = _mm512_sub_pd(s_para_d, p_para_d);    
                _mm512_store_pd((sum + tmp_s_bias), new_s_para_d);                            
            }     
            len[a_old[tmp_index]] -= 1;

            self_d[tmp_index] = 0;
            __m512 sq_para_d = _mm512_set1_ps(0);
            for (int par_d = 0; par_d < d / para_d; par_d++){
                int tmp_c_bias = a_old[tmp_index] * d + par_d * para_d;
                long long int tmp_p_bias = 1l * tmp_index * d + par_d * para_d;
                __m512 p_para_d = _mm512_load_ps((dmatrix + tmp_p_bias));
                __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias)); 
                __m512 sub_para_d = _mm512_sub_ps(p_para_d, c_para_d);
                sq_para_d = _mm512_fmadd_ps(sub_para_d, sub_para_d, sq_para_d);               
            }
            self_d[tmp_index] = sqrt(_mm512_reduce_add_ps(sq_para_d));

            for (int j = 0; j < k; j++){
                long long int tmp_index_l = 1l * tmp_index * k + j;
                if (j != a[tmp_index]){
                    if (l[tmp_index_l] < self_d[tmp_index]){

                        __m512 sq_para_d = _mm512_set1_ps(0);
                        for (int par_d = 0; par_d < d / para_d; par_d++){
                            int tmp_c_bias = j * d + par_d * para_d;
                            long long int tmp_p_bias = 1l * tmp_index * d + par_d * para_d;
                            __m512 p_para_d = _mm512_load_ps((dmatrix + tmp_p_bias));
                            __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias)); 
                            __m512 sub_para_d = _mm512_sub_ps(p_para_d, c_para_d);
                            sq_para_d = _mm512_fmadd_ps(sub_para_d, sub_para_d, sq_para_d);                   
                        }
                        l[tmp_index_l] = sqrt(_mm512_reduce_add_ps(sq_para_d));

                        if (l[tmp_index_l] < self_d[tmp_index]){
                            a[tmp_index] = j;
                            self_d[tmp_index] = l[tmp_index_l];
                        }
                    }
                }
            }

            sse[a[tmp_index]] += self_d[tmp_index] * self_d[tmp_index];
            len[a[tmp_index]] += 1;   

            for (int j = 0; j < d / para_d_db; j++){
                long long int tmp_p_bias = 1l * tmp_index * d + j * para_d_db;
                int tmp_s_bias = a[tmp_index] * d + j * para_d_db;

                __m256 p_para_d_f = _mm256_load_ps((dmatrix + tmp_p_bias)); 
                float *p_para_d_f_p = (float *)&p_para_d_f;    
                __m512d p_para_d = _mm512_set_pd(*(p_para_d_f_p + 7), \
                                                 *(p_para_d_f_p + 6), \
                                                 *(p_para_d_f_p + 5), \
                                                 *(p_para_d_f_p + 4), \
                                                 *(p_para_d_f_p + 3), \
                                                 *(p_para_d_f_p + 2), \
                                                 *(p_para_d_f_p + 1), \
                                                 *(p_para_d_f_p + 0));
                __m512d s_para_d = _mm512_load_pd((sum + tmp_s_bias));     
                __m512d new_s_para_d = _mm512_add_pd(s_para_d, p_para_d);    
                _mm512_store_pd((sum + tmp_s_bias), new_s_para_d);                            
            }      
        }

        for (int i = 0; i < M1subM0_pt; i++){
            long long int tmp_index = M0 + 1l * par * M1subM0_pt + i;
            for (int j = 0; j < k; j++){
                long long int tmp_index_l = 1l * tmp_index * k + j;
                __m512 sq_para_d = _mm512_set1_ps(0);
                for (int par_d = 0; par_d < d / para_d; par_d++){
                    int tmp_c_bias = j * d + par_d * para_d;
                    long long int tmp_p_bias = 1l * tmp_index * d + par_d * para_d;
                    __m512 p_para_d = _mm512_load_ps((dmatrix + tmp_p_bias));
                    __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias)); 
                    __m512 sub_para_d = _mm512_sub_ps(p_para_d, c_para_d);
                    sq_para_d = _mm512_fmadd_ps(sub_para_d, sub_para_d, sq_para_d);                   
                }
                l[tmp_index_l] = sqrt(_mm512_reduce_add_ps(sq_para_d));
            }

            int min_index = 0;
            float min_l = INFINITY;
            for (int j = 0; j < k; j++){
                long long int tmp_index_l = 1l * tmp_index * k + j;
                if (l[tmp_index_l] < min_l){
                    min_l = l[tmp_index_l];
                    min_index = j;
                }
            }
            a[tmp_index] = min_index;
            self_d[tmp_index] = l[tmp_index * k + a[tmp_index]];
            sse[a[tmp_index]] += self_d[tmp_index] * self_d[tmp_index];
            len[a[tmp_index]]++;          
  
            for (int j = 0; j < d / para_d_db; j++){
                long long int tmp_p_bias = 1l * tmp_index * d + j * para_d_db;
                int tmp_s_bias = a[tmp_index] * d + j * para_d_db;

                __m256 p_para_d_f = _mm256_load_ps((dmatrix + tmp_p_bias)); 
                float *p_para_d_f_p = (float *)&p_para_d_f;    
                __m512d p_para_d = _mm512_set_pd(*(p_para_d_f_p + 7), \
                                                 *(p_para_d_f_p + 6), \
                                                 *(p_para_d_f_p + 5), \
                                                 *(p_para_d_f_p + 4), \
                                                 *(p_para_d_f_p + 3), \
                                                 *(p_para_d_f_p + 2), \
                                                 *(p_para_d_f_p + 1), \
                                                 *(p_para_d_f_p + 0));
                __m512d s_para_d = _mm512_load_pd((sum + tmp_s_bias));     
                __m512d new_s_para_d = _mm512_add_pd(s_para_d, p_para_d);    
                _mm512_store_pd((sum + tmp_s_bias), new_s_para_d);                            
            }  
        }
        // if(iter_index == 1)  printf("[par %d] check point 0.3 \n", par);
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

void *par_task_srmbkm(void *info){
    // pthread_detach(pthread_self());
    ptinfo ptinfo_par = *(ptinfo *)info;
    int batchsize_pt = ptinfo_par.batchsize_pt;
    int batchsize = ptinfo_par.batchsize;
    int k = ptinfo_par.k;
    int d = ptinfo_par.d;
    double *len = ptinfo_par.len;
    double *sum = ptinfo_par.sum;
    double *lenE = ptinfo_par.lenE;
    double *sumE = ptinfo_par.sumE;
    float *dmatrix = ptinfo_par.dmatrix;
    float *means = ptinfo_par.means;
    int *rindex = ptinfo_par.rindex;
    int par = ptinfo_par.par;
    int batch_num = ptinfo_par.batch_num;
    int iteration = ptinfo_par.iteration;
    double alpha = ptinfo_par.alpha;

    for (int h = 0; h < iteration; h++){
        pthread_barrier_wait(&barrier);

        int bindex = h % batch_num;

        for (int i = 0; i < batchsize_pt; i++) {
            float min_dist = FLT_MAX;
            int index = 0;
            for (int j = 0; j < k; j++) {
                __m512 sq_para_d = _mm512_set1_ps(0);
                for (int m = 0; m < d / para_d; m++){
                    long long int tmp_p_bias = 1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + m * para_d;
                    int tmp_c_bias = j * d + m * para_d;
                    __m512 p_para_d = _mm512_load_ps((dmatrix + tmp_p_bias));
                    __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias)); 
                    __m512 sub_para_d = _mm512_sub_ps(p_para_d, c_para_d);
                    sq_para_d = _mm512_fmadd_ps(sub_para_d, sub_para_d, sq_para_d);
                }

                float dist = _mm512_reduce_add_ps(sq_para_d);

                if (dist < min_dist) {
                    min_dist = dist;
                    index = j;
                }
            }
            
            len[index]++;
            lenE[index]++;
            for (int j = 0; j < d / para_d_db; j++){
                long long int tmp_p_bias = 1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + j * para_d_db;
                int tmp_s_bias = index * d + j * para_d_db;
                __m256 p_para_d_f = _mm256_load_ps((dmatrix + tmp_p_bias)); 
                float *p_para_d_f_p = (float *)&p_para_d_f;    
                __m512d p_para_d = _mm512_set_pd(*(p_para_d_f_p + 7), \
                                                 *(p_para_d_f_p + 6), \
                                                 *(p_para_d_f_p + 5), \
                                                 *(p_para_d_f_p + 4), \
                                                 *(p_para_d_f_p + 3), \
                                                 *(p_para_d_f_p + 2), \
                                                 *(p_para_d_f_p + 1), \
                                                 *(p_para_d_f_p + 0));
                __m512d s_para_d = _mm512_load_pd((sum + tmp_s_bias));  
                __m512d se_para_d = _mm512_load_pd((sumE + tmp_s_bias));    
                __m512d new_s_para_d = _mm512_add_pd(p_para_d, s_para_d);   
                __m512d new_se_para_d = _mm512_add_pd(p_para_d, se_para_d);   
                _mm512_store_pd((sum + tmp_s_bias), new_s_para_d);   
                _mm512_store_pd((sumE + tmp_s_bias), new_se_para_d);                           
            }
        }

//------------------------------------------
        // int bindex = h % batch_num;
        // // if(par == 0 && h < 10){
        // //     printf("bindex = %d\n", bindex);
        // // }
        // for (int i = 0; i < batchsize_pt; i++) {
        //     float min_dist = FLT_MAX;
        //     int index = 0;
        //     for (int j = 0; j < k; j++) {
        //         float dist = 0;
        //         for (int m = 0; m < d; m++){
        //             float coor_1, coor_2;
        //             // if(i == 0 && alpha == 0.01){
        //             //     printf("par=%d, dmatrix[%d]\n", par, 1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + m);
        //             // }
        //             coor_1 = dmatrix[1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + m];
        //             coor_2 = means[j * d + m];
        //             dist += (coor_1 - coor_2) * (coor_1 - coor_2);
        //             //dist += 0.5*coor_2*coor_2 - coor_1*coor_2;
        //         }
        //         if (dist < min_dist) { /* find the min and its array index */
        //             min_dist = dist;
        //             index = j;
        //         }
        //     }
        //     len[index]++;
        //     lenE[index]++;
        //     for (int j = 0; j < d; j++){
        //         float a = dmatrix[1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + j];
        //         sum[index * d + j] += dmatrix[1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + j];
        //         sumE[index * d + j] += dmatrix[1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + j];
        //     }
        // }
        pthread_barrier_wait(&barrier);  
    }
    pthread_exit(NULL);
}

void *par_task_srnmbkm(void *info){
    ptinfo ptinfo_par = *(ptinfo *)info;
    int k = ptinfo_par.k;
    int d = ptinfo_par.d;
    double *len = ptinfo_par.len;
    double *sum = ptinfo_par.sum;
    double *lenE = ptinfo_par.lenE;
    double *sumE = ptinfo_par.sumE;
    float *dmatrix = ptinfo_par.dmatrix;
    float *means = ptinfo_par.means;
    int par = ptinfo_par.par;
    int iteration = ptinfo_par.iteration;
    int mbatchsize_pt = ptinfo_par.mbatchsize_pt;
    int mbatchsize = ptinfo_par.mbatchsize;
    float *sse = ptinfo_par.sse;
    int *M1 = ptinfo_par.M1;

    int iter_index = ptinfo_par.iter_index;
    // if(par == 0)  printf("[par %d] mbatchsize_pt:%d \n", par, mbatchsize_pt);

    for (int h = 0; h < iteration; h++){       
        pthread_barrier_wait(&barrier);
        for (int m = 0; m < *M1 / mbatchsize; m++) {
            //if(par == 0) printf("checkpoint-0_task\n");
            pthread_barrier_wait(&barrier);
            for (int i = 0; i < mbatchsize_pt; i++) {
                // if(par == 0)  printf("[par %d]i:%d \n", par, i);
                float min_dist = INFINITY;
                int index = 0;
                for (int j = 0; j < k; j++) {

                    __m512 sq_para_d = _mm512_set1_ps(0);
                    for (int par_d = 0; par_d < d / para_d; par_d++) {
                        long long int tmp_p_bias = 1l * m * mbatchsize * d + 1l * par * mbatchsize_pt * d + 1l * i * d + par_d * para_d;
                        int tmp_c_bias = j * d + par_d * para_d;
                        __m512 p_para_d = _mm512_load_ps((dmatrix + tmp_p_bias));
                        __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias)); 
                        __m512 sub_para_d = _mm512_sub_ps(p_para_d, c_para_d);
                        sq_para_d = _mm512_fmadd_ps(sub_para_d, sub_para_d, sq_para_d);                        
                    }
                    float dist = _mm512_reduce_add_ps(sq_para_d);

                    if (dist < min_dist) { /* find the min and its array index */
                        min_dist = dist;
                        index = j;
                    }
                }

                len[index]++;
                lenE[index]++;

                for (int j = 0; j < d / para_d_db; j++){
                    long long int tmp_p_bias = 1l * m * mbatchsize * d + 1l * par * mbatchsize_pt * d + 1l * i * d + j * para_d_db;
                    int tmp_s_bias = index * d + j * para_d_db;
                    __m256 p_para_d_f = _mm256_load_ps((dmatrix + tmp_p_bias)); 
                    float *p_para_d_f_p = (float *)&p_para_d_f;    
                    __m512d p_para_d = _mm512_set_pd(*(p_para_d_f_p + 7), \
                                                     *(p_para_d_f_p + 6), \
                                                     *(p_para_d_f_p + 5), \
                                                     *(p_para_d_f_p + 4), \
                                                     *(p_para_d_f_p + 3), \
                                                     *(p_para_d_f_p + 2), \
                                                     *(p_para_d_f_p + 1), \
                                                     *(p_para_d_f_p + 0));
                    __m512d s_para_d = _mm512_load_pd((sum + tmp_s_bias));
                    __m512d se_para_d = _mm512_load_pd((sumE + tmp_s_bias));    
                    __m512d new_s_para_d = _mm512_add_pd(p_para_d, s_para_d);
                    __m512d new_se_para_d = _mm512_add_pd(p_para_d, se_para_d);   
                    _mm512_store_pd((sum + tmp_s_bias), new_s_para_d);
                    _mm512_store_pd((sumE + tmp_s_bias), new_se_para_d);                      
                }
                // if(par == 0)  printf("[par %d]here-1\n", par);
                sse[index] += min_dist;
            }  
            pthread_barrier_wait(&barrier);
            //if(par == 0) printf("checkpoint-1_task\n");
        }
        pthread_barrier_wait(&barrier);
        //pthread_barrier_wait(&barrier); 
    }
    pthread_exit(NULL);
}

double cal_loss(int n, int d, int k, FILE *fp, float *dmatrix, float *means, int threads){

    double *loss = (double*) ddr_alloc(threads * sizeof (double)); 
    memset(loss, 0 , threads * sizeof (double)); 

    // 批样本大小batch size,仅用所有数据能容纳的batch size个样本来做K-Means聚类，其他数据丢
    int batchsize = n;
    int batch_num = 1;

    pthread_t Thread[NUM_THREADS];
    ptinfo ptinfo_arr[NUM_THREADS];
    int batchsize_pt = batchsize / threads;
	for(int par = 0; par < threads; par++){
        ptinfo_arr[par].dmatrix = dmatrix;
        ptinfo_arr[par].means = means;
        ptinfo_arr[par].batchsize_pt = batchsize_pt;
        ptinfo_arr[par].d = d;
        ptinfo_arr[par].k = k;
        ptinfo_arr[par].par = par;
        ptinfo_arr[par].iteration = 1;
        ptinfo_arr[par].loss = loss + par;
		pthread_create(&Thread[par], NULL, par_task_loss, ptinfo_arr + par);
	} 
    
    pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);
    
    double loss_total = 0;
    for (int par = 0; par < threads; par++) {
        loss_total += loss[par];
    }  
    free(loss);

    for(int par = 0; par < threads; par++){
		pthread_join(Thread[par], NULL);
	} 

    return loss_total;
}

void run_nmbkm(int n, int d, int k, int iteration, FILE *fp, float *dmatrix, float *means, 
                int batchsize, int seed, int threads,
                double **len, double **sum, float **sse, float *p, float *l, 
                int *a_old, int *a, float *self_d, float *means_old, float *sigmaC, 
                int *M0, int *M1, int *data_amount, float *accu_runtime, float rho,
                int iter_index){

    pthread_t Thread[NUM_THREADS];
    ptinfo ptinfo_arr[NUM_THREADS];
	for(int par = 0; par < threads; par++){
		ptinfo_arr[par].len = len[par];
        ptinfo_arr[par].sum = sum[par];
        ptinfo_arr[par].dmatrix = dmatrix;
        ptinfo_arr[par].means = means;
        ptinfo_arr[par].d = d;
        ptinfo_arr[par].k = k;
        ptinfo_arr[par].par = par;
        ptinfo_arr[par].iteration = iteration;
        ptinfo_arr[par].M0 = M0;
        ptinfo_arr[par].M1 = M1;
        ptinfo_arr[par].sse = sse[par];
        ptinfo_arr[par].p = p;
        ptinfo_arr[par].l = l;
        ptinfo_arr[par].a_old = a_old;
        ptinfo_arr[par].a = a;
        ptinfo_arr[par].self_d = self_d;
        ptinfo_arr[par].means_old = means_old;
        ptinfo_arr[par].sigmaC = sigmaC;
        ptinfo_arr[par].iter_index = iter_index;
        ptinfo_arr[par].threads = threads;

		pthread_create(&Thread[par], NULL, par_task_nmbkm, &ptinfo_arr[par]);
	}

    CUtilTimer timer;
    for (int h = 0; h < iteration; h++){
        timer.start();

        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);

        for (int j = 0; j < k; j++){
            for (int par_d = 0; par_d < d / para_d; par_d++){
                int tmp_c_bias = j * d + par_d * para_d;
                __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias)); 
                _mm512_store_ps((means_old + tmp_c_bias), c_para_d);                   
            }
        }

        for (int j = 0; j < k; j++){
            double tmp_len = 0;
            double tmp_sse = 0;
            for (int par = 0; par < threads; par++) {
                tmp_len += len[par][j];    
                tmp_sse += sse[par][j];   
            }
            sigmaC[j] = sqrt(tmp_sse / (tmp_len * (tmp_len - 1)));
            // printf("[%d]tmp_sse:%f, tmp_len:%f\n", j, tmp_sse, tmp_len);
   
            if (tmp_len > 0){
                for (int par_d = 0; par_d < d / para_d_db; par_d++){
                    int tmp_s_bias = j * d + par_d * para_d_db;
                    __m512d tmp_s_para_d = _mm512_set1_pd(0);
                    for (int par = 0; par < threads; par++) {
                        __m512d s_para_d = _mm512_load_pd((sum[par] + tmp_s_bias));
                        tmp_s_para_d = _mm512_add_pd(s_para_d, tmp_s_para_d);
                    }
                    __m512d v_para_d = _mm512_set1_pd(tmp_len);
                    __m512d new_c_para_d = _mm512_div_pd(tmp_s_para_d, v_para_d);
                    double *new_c_para_d_p = (double *)&new_c_para_d;    
                    __m256 new_c_para_d_f = _mm256_set_ps(*(new_c_para_d_p + 7), \
                                                     *(new_c_para_d_p + 6), \
                                                     *(new_c_para_d_p + 5), \
                                                     *(new_c_para_d_p + 4), \
                                                     *(new_c_para_d_p + 3), \
                                                     *(new_c_para_d_p + 2), \
                                                     *(new_c_para_d_p + 1), \
                                                     *(new_c_para_d_p + 0));                   
                    _mm256_store_ps(means + tmp_s_bias, new_c_para_d_f);
                }   
            }     

            p[j] = 0;
            __m512 sq_para_d = _mm512_set1_ps(0);
            for (int par_d = 0; par_d < d / para_d; par_d++){
                int tmp_c_bias = j * d + par_d * para_d;
                __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias));
                __m512 co_para_d = _mm512_load_ps((means_old + tmp_c_bias)); 
                __m512 sub_para_d = _mm512_sub_ps(c_para_d, co_para_d);
                sq_para_d = _mm512_fmadd_ps(sub_para_d, sub_para_d, sq_para_d);                   
            }
            p[j] = sqrt(_mm512_reduce_add_ps(sq_para_d));
        }
        
        float min_rho = INFINITY;
        for (int j = 0; j < k; j++){
            float tmp_rho = sigmaC[j] / p[j];
            // printf("sigmaC:%f, p:%f\n", sigmaC[j], p[j]);
            if (!isnan(tmp_rho) && tmp_rho < min_rho){
                min_rho = tmp_rho;
            }
        }
        int M_old = *M0;
        if (min_rho > rho){
            *M0 = *M1;
            *M1 = (2 * *M1) < n ? (2 * *M1) : n;
        }
        else{
            *M0 = *M1;
        }
        timer.stop();
        *accu_runtime += timer.get_time();
        *data_amount += *M0;
    }

    // for(int i = 0; i < 16; i++){
    //     printf("%d\t", a[i]);
    // }
    // printf("\n");

    for(int par = 0; par < threads; par++){
		pthread_join(Thread[par], NULL);
	} 
}

void run_srnmbkm(int n, int d, int k, int iteration, FILE *fp, float *dmatrix, float *means, 
                int batchsize, int seed, int threads,
                double **len, double **lenE, double **sum, double **sumE, float **sse, float *p, float *l, 
                int *a_old, int *a, float *self_d, float *means_old, float *sigmaC, 
                int *M0, int *M1, int *data_amount, float *accu_runtime, float rho, int mbatchsize, int *epoch_count,
                float alpha, int iter_index){

    pthread_t Thread[NUM_THREADS];
    ptinfo ptinfo_arr[NUM_THREADS];
	for(int par = 0; par < threads; par++){
		ptinfo_arr[par].len = len[par];
        ptinfo_arr[par].sum = sum[par];
        ptinfo_arr[par].lenE = lenE[par];
        ptinfo_arr[par].sumE = sumE[par];
        ptinfo_arr[par].dmatrix = dmatrix;
        ptinfo_arr[par].means = means;
        ptinfo_arr[par].iteration = iteration;
        ptinfo_arr[par].M1 = M1;
        ptinfo_arr[par].d = d;
        ptinfo_arr[par].k = k;
        ptinfo_arr[par].par = par;
        ptinfo_arr[par].sse = sse[par];
        ptinfo_arr[par].mbatchsize = mbatchsize;
        ptinfo_arr[par].mbatchsize_pt = mbatchsize / threads;
        ptinfo_arr[par].iter_index = iter_index;

		pthread_create(&Thread[par], NULL, par_task_srnmbkm, &ptinfo_arr[par]);
	}

    CUtilTimer timer;
    for (int h = 0; h < iteration; h++){
        
        timer.start();
        for (int j = 0; j < k; j++){
            for (int par_d = 0; par_d < d / para_d; par_d++){
                int tmp_c_bias = j * d + par_d * para_d;
                __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias)); 
                _mm512_store_ps((means_old + tmp_c_bias), c_para_d);                   
            }
        }
        //printf("checkpoint-2\n");
        pthread_barrier_wait(&barrier);
        for (int m = 0; m < *M1 / mbatchsize; m++) {

            //printf("checkpoint-0\n");
            pthread_barrier_wait(&barrier);
            pthread_barrier_wait(&barrier);
            //printf("checkpoint-1\n");
            

            for (int i = 0; i < k; i++) {
                double tmp_len = 0;
                for (int par = 0; par < threads; par++) {
                    tmp_len += len[par][i];    
                    
                }
                // printf("%d \t", tmp_len);
                if (tmp_len > 0){
                    for (int par_d = 0; par_d < d / para_d_db; par_d++){
                        int tmp_s_bias = i * d + par_d * para_d_db;
                        __m512d tmp_s_para_d = _mm512_set1_pd(0);
                        for (int par = 0; par < threads; par++) {
                            __m512d s_para_d = _mm512_load_pd((sum[par] + tmp_s_bias));
                            tmp_s_para_d = _mm512_add_pd(s_para_d, tmp_s_para_d);
                        }
                        __m512d v_para_d = _mm512_set1_pd(tmp_len);
                        __m512d new_c_para_d = _mm512_div_pd(tmp_s_para_d, v_para_d);
                        double *new_c_para_d_p = (double *)&new_c_para_d;    
                        __m256 new_c_para_d_f = _mm256_set_ps(*(new_c_para_d_p + 7), \
                                                         *(new_c_para_d_p + 6), \
                                                         *(new_c_para_d_p + 5), \
                                                         *(new_c_para_d_p + 4), \
                                                         *(new_c_para_d_p + 3), \
                                                         *(new_c_para_d_p + 2), \
                                                         *(new_c_para_d_p + 1), \
                                                         *(new_c_para_d_p + 0));                   
                        _mm256_store_ps(means + tmp_s_bias, new_c_para_d_f);
                    }   
                }
            }
            // printf("\n");
        }    
        pthread_barrier_wait(&barrier);   
        //printf("checkpoint-2\n"); 

        for (int j = 0; j < k; j++){
            double tmp_sse = 0;
            double tmp_lenE = 0;
            for (int par = 0; par < threads; par++) {
                tmp_sse += sse[par][j];   
                tmp_lenE += lenE[par][j];  
            }            
            sigmaC[j] = sqrt(tmp_sse / (tmp_lenE * (tmp_lenE - 1)));
            // printf("[%d]tmp_sse:%f, tmp_lenE:%f\n", j, tmp_sse, tmp_lenE);
            p[j] = 0;
            __m512 sq_para_d = _mm512_set1_ps(0);
            for (int par_d = 0; par_d < d / para_d; par_d++){
                int tmp_c_bias = j * d + par_d * para_d;
                __m512 p_para_d = _mm512_load_ps((means + tmp_c_bias));
                __m512 po_para_d = _mm512_load_ps((means_old + tmp_c_bias)); 
                __m512 sub_para_d = _mm512_sub_ps(p_para_d, po_para_d);
                sq_para_d = _mm512_fmadd_ps(sub_para_d, sub_para_d, sq_para_d);                   
            }
            p[j] = sqrt(_mm512_reduce_add_ps(sq_para_d));
        }

        float min_rho = INFINITY;
        for (int j = 0; j < k; j++){
            float tmp_rho = sigmaC[j] / p[j];
            // printf("sigmaC:%f, p:%f\n", sigmaC[j], p[j]);
            if (!isnan(tmp_rho) && tmp_rho < min_rho){
                min_rho = tmp_rho;
            }
        }

        // printf("min_rho = %f\n", min_rho);

        int M_old = *M1;
        if (min_rho > rho && *M1 != n){
	
            *M1 = (2 * *M1) < n ? (2 * *M1) : n;
            *epoch_count = 0;
            for (int par = 0; par < threads; par++) {
                memset(len[par], 0 , k * sizeof (double));  
                memset(lenE[par], 0 , k * sizeof (double));  
                memset(sum[par], 0 , k * d * sizeof (double));  
                memset(sumE[par], 0 , k * d * sizeof (double));   
                memset(sse[par], 0 , k * sizeof (float));              
            }   
        }
        else{
            *epoch_count = *epoch_count + 1;
            double coefficient = *epoch_count * alpha;
             for (int i = 0; i < k; i++) {
                double tmp_lenE = 0;
                for (int par = 0; par < threads; par++) {
                    tmp_lenE += lenE[par][i];    
                    len[par][i] = lenE[par][i] * coefficient;
                }
                
                for (int par_d = 0; par_d < d / para_d_db; par_d++){
                    __m512d se_para_d = _mm512_set1_pd(0);
                    int tmp_se_bias = i * d + par_d * para_d_db;
                    for (int par = 0; par < threads; par++) {
                        __m512d tmp_se_para_d = _mm512_load_pd((sumE[par] + tmp_se_bias));
                        se_para_d = _mm512_add_pd(se_para_d, tmp_se_para_d);

                        __m512d coeff_para_d = _mm512_set1_pd(coefficient);
                        __m512d tmp_s_para_d = _mm512_mul_pd(tmp_se_para_d, coeff_para_d);
                        _mm512_store_pd(sum[par] + tmp_se_bias, tmp_s_para_d);
                    }

                    if(tmp_lenE > 0){
                        __m512d v_para_d = _mm512_set1_pd(tmp_lenE);
                        __m512d new_c_para_d = _mm512_div_pd(se_para_d, v_para_d);
                        double *new_c_para_d_p = (double *)&new_c_para_d;    
                        __m256 new_c_para_d_f = _mm256_set_ps(*(new_c_para_d_p + 7), \
                                                         *(new_c_para_d_p + 6), \
                                                         *(new_c_para_d_p + 5), \
                                                         *(new_c_para_d_p + 4), \
                                                         *(new_c_para_d_p + 3), \
                                                         *(new_c_para_d_p + 2), \
                                                         *(new_c_para_d_p + 1), \
                                                         *(new_c_para_d_p + 0));                   
                        _mm256_store_ps(means + tmp_se_bias, new_c_para_d_f); 
                    }
                }
            }
            for (int par = 0; par < threads; par++) {
                memset(lenE[par], 0 , k * sizeof (double));  
                memset(sumE[par], 0 , k * d * sizeof (double));   
                memset(sse[par], 0 , k * sizeof (float));              
            }  

        }
        //if(h % batch_num == batch_num - 1){
        timer.stop();
        *accu_runtime += timer.get_time();
        *data_amount += M_old;
        //}
        //pthread_barrier_wait(&barrier);
        //printf("checkpoint-3\n"); 
    }
    for(int par = 0; par < threads; par++){
	    pthread_join(Thread[par], NULL);
	}
    //printf("checkpoint-4\n"); 
}

void start_nmbkm(int n, int d, int k, char *filename, float *dmatrix, float *means, int batchsize, int seed, int threads){
    char fout[100];
    sprintf(fout, "%s/nmbkm/%s_k%d_sd%d_bs%d.txt", \
            fout_root, filename, k, seed, batchsize);
    FILE *fp;
    if((fp=fopen(fout,"wt+")) == NULL){
        printf("Cannot open %s!", fout);
        exit(1);
    }

    double **len = (double **) ddr_alloc(threads * sizeof (double *)); 
    double **sum = (double **) ddr_alloc(threads * sizeof (double *));
    float **sse = (float **) ddr_alloc(k * sizeof (float *));
    for (int par = 0; par < threads; par++) {
        len[par] = (double*) ddr_alloc(k * sizeof (double)); 
        sum[par] = (double*) ddr_alloc(k * d * sizeof (double));
        sse[par] = (float*) ddr_alloc(k * sizeof (float));
        memset(len[par], 0 , k * sizeof (double));   
        memset(sum[par], 0 , k * d * sizeof (double));      
        memset(sse[par], 0 , k * sizeof (float));     
    }    
    
    float *p = (float*) ddr_alloc(k * sizeof (float));
    float *l = (float*) ddr_alloc(1l * n * k * sizeof (float));   
    int *a_old = (int*) ddr_alloc(n * sizeof (int));  
    int *a = (int*) ddr_alloc(n * sizeof (int));

    float *self_d = (float*) ddr_alloc(n * sizeof (float));
    float *means_old = (float*) ddr_alloc(k * d * sizeof (float));   
    float *sigmaC = (float*) ddr_alloc(k * sizeof (float));
     
    memset(p, 0 , k * sizeof (float));

    int *M0 = (int*) ddr_alloc(sizeof (int));
    int *M1 = (int*) ddr_alloc(sizeof (int));
    int *data_amount = (int*) ddr_alloc(sizeof (int));
    float *accu_runtime = (float*) ddr_alloc(sizeof (float));
    *M0 = 0;
    *M1 = batchsize;
    *data_amount = 0;
    *accu_runtime = 0;

    int iteration = 1000;
    float rho = 10;

    printf("iteration: %d; n: %d; d: %d; k: %d; batchsize: %d; seed: %d; rho: %f\n", \
            iteration, n, d, k, batchsize, seed, rho); 

    double loss = cal_loss(n, d, k, fp, dmatrix, means, threads);
    fprintf(fp, "initial loss: %lf\n", loss);
    for(int i = 0; i < iteration; i++){
        run_nmbkm(n, d, k, 1, fp, dmatrix, means, batchsize, seed, threads,
                len, sum, sse, p, l, a_old, a, self_d, means_old, sigmaC, 
                M0, M1, data_amount, accu_runtime, rho, i);
        if((i < 9) || (i % 10 == 9)){
            loss = cal_loss(n, d, k, fp, dmatrix, means, threads);
            fprintf(fp,"iteration:%d, data:%d, time:%f, loss: %lf\n", \
            i + 1, *data_amount, *accu_runtime, loss);  
            printf("iteration:%d, data:%d, time:%f, loss: %lf\n",  \
            i + 1, *data_amount, *accu_runtime, loss);
        }
    }
    fclose(fp);
    for (int par = 0; par < threads; par++) {
        free(len[par]);   
        free(sum[par]);   
        free(sse[par]);        
    }
    free(len);  
    free(sum);  
    free(p);
    free(l);
    free(a_old);
    free(a);
    free(self_d);    
    free(means_old);
    free(sigmaC);
    free(M0);
    free(M1);
    free(data_amount);
    free(accu_runtime);
}

void run_srmbkm(int n, int d, int k, int epoch_num, FILE *fp, float *dmatrix, float *means, \
            int batchsize, int seed, double alpha, int threads, int epoch_count,
            double **len, double **lenE, double **sum, double **sumE, float *accu_runtime){

    int batch_num = n / batchsize;

    CUtilTimer timer;

    pthread_t Thread[NUM_THREADS];
    ptinfo ptinfo_arr[NUM_THREADS];
    int batchsize_pt = batchsize / threads;
	for(int par = 0; par < threads; par++){
		ptinfo_arr[par].len = len[par];
        ptinfo_arr[par].sum = sum[par];
		ptinfo_arr[par].lenE = lenE[par];
        ptinfo_arr[par].sumE = sumE[par];
        ptinfo_arr[par].dmatrix = dmatrix;
        ptinfo_arr[par].means = means;
        ptinfo_arr[par].batchsize_pt = batchsize_pt;
        ptinfo_arr[par].batchsize = batchsize;
        ptinfo_arr[par].d = d;
        ptinfo_arr[par].k = k;
        ptinfo_arr[par].par = par;
        ptinfo_arr[par].batch_num = batch_num;
        ptinfo_arr[par].iteration = epoch_num * batch_num;
		pthread_create(&Thread[par], NULL, par_task_srmbkm, &ptinfo_arr[par]);
	}

    // srand(seed);
    // int epoch_count = 0;
    double coefficient = 0;
    int printf_flag = 0;

    for (int h = 0; h < epoch_num * batch_num; h++){
        timer.start();

        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);  
    

        for (int i = 0; i < k; i++) {
            double tmp_len = 0;
            for (int par = 0; par < threads; par++) {
                tmp_len += len[par][i];    
            }
            if (tmp_len > 0){
                for (int par_d = 0; par_d < d / para_d_db; par_d++){
                    int tmp_s_bias = i * d + par_d * para_d_db;
                    __m512d tmp_s_para_d = _mm512_set1_pd(0);
                    for (int par = 0; par < threads; par++) {
                        __m512d s_para_d = _mm512_load_pd((sum[par] + tmp_s_bias));
                        tmp_s_para_d = _mm512_add_pd(s_para_d, tmp_s_para_d);
                    }
                    __m512d v_para_d = _mm512_set1_pd(tmp_len);
                    __m512d new_c_para_d = _mm512_div_pd(tmp_s_para_d, v_para_d);
                    double *new_c_para_d_p = (double *)&new_c_para_d;    
                    __m256 new_c_para_d_f = _mm256_set_ps(*(new_c_para_d_p + 7), \
                                                     *(new_c_para_d_p + 6), \
                                                     *(new_c_para_d_p + 5), \
                                                     *(new_c_para_d_p + 4), \
                                                     *(new_c_para_d_p + 3), \
                                                     *(new_c_para_d_p + 2), \
                                                     *(new_c_para_d_p + 1), \
                                                     *(new_c_para_d_p + 0));                   
                    _mm256_store_ps(means + tmp_s_bias, new_c_para_d_f);
                }   
            }
        }   
        timer.stop();
        *accu_runtime += timer.get_time();
        
        if(h % batch_num == batch_num - 1){
            timer.start();
            printf_flag = 1;
            coefficient = (epoch_count + 1) * alpha; 

            for (int i = 0; i < k; i++) {
                int tmp_lenE = 0;
                for (int par = 0; par < threads; par++) {
                    tmp_lenE += lenE[par][i];    
                    len[par][i] = lenE[par][i] * coefficient;
                }
                
                for (int par_d = 0; par_d < d / para_d_db; par_d++){
                    __m512d se_para_d = _mm512_set1_pd(0);
                    int tmp_se_bias = i * d + par_d * para_d_db;
                    for (int par = 0; par < threads; par++) {
                        __m512d tmp_se_para_d = _mm512_load_pd((sumE[par] + tmp_se_bias));
                        se_para_d = _mm512_add_pd(se_para_d, tmp_se_para_d);

                        __m512d coeff_para_d = _mm512_set1_pd(coefficient);
                        __m512d tmp_s_para_d = _mm512_mul_pd(tmp_se_para_d, coeff_para_d);
                        _mm512_store_pd(sum[par] + tmp_se_bias, tmp_s_para_d);
                    }

                    if(tmp_lenE > 0){
                        __m512d v_para_d = _mm512_set1_pd(tmp_lenE);
                        __m512d new_c_para_d = _mm512_div_pd(se_para_d, v_para_d);
                        double *new_c_para_d_p = (double *)&new_c_para_d;    
                        __m256 new_c_para_d_f = _mm256_set_ps(*(new_c_para_d_p + 7), \
                                                         *(new_c_para_d_p + 6), \
                                                         *(new_c_para_d_p + 5), \
                                                         *(new_c_para_d_p + 4), \
                                                         *(new_c_para_d_p + 3), \
                                                         *(new_c_para_d_p + 2), \
                                                         *(new_c_para_d_p + 1), \
                                                         *(new_c_para_d_p + 0));                   
                        _mm256_store_ps(means + tmp_se_bias, new_c_para_d_f); 
                    }
                }
            }
            
            for (int par = 0; par < threads; par++) {
                memset(lenE[par], 0 , k * sizeof (double));
                memset(sumE[par], 0 , k * d * sizeof (double));
            } 
            timer.stop();
            *accu_runtime += timer.get_time();
        }
    }

    for(int par = 0; par < threads; par++){
		pthread_join(Thread[par], NULL);
	} 

}

void start_srnmbkm(int n, int d, int k, char *filename, float *dmatrix, float *means, int batchsize, int seed, float alpha, int threads, int mbatchsize){

    char fout[100];
    sprintf(fout, "%s/srnmbkm/%s_k%d_sd%d_bs%d_mbs%d_a%f.txt", \
            fout_root, filename, k, seed, batchsize, mbatchsize, alpha);
    FILE *fp;
    if((fp=fopen(fout,"wt+")) == NULL){
        printf("Cannot open %s!", fout);
        exit(1);
    }

    double **len = (double **) ddr_alloc(threads * sizeof (double *)); 
    double **lenE = (double **) ddr_alloc(threads * sizeof (double *)); 
    double **sum = (double **) ddr_alloc(threads * sizeof (double *));
    double **sumE = (double **) ddr_alloc(threads * sizeof (double *));
    float **sse = (float **) ddr_alloc(k * sizeof (float *));
    for (int par = 0; par < threads; par++) {
        len[par] = (double*) ddr_alloc(k * sizeof (double)); 
        lenE[par] = (double*) ddr_alloc(k * sizeof (double)); 
        sum[par] = (double*) ddr_alloc(k * d * sizeof (double));
        sumE[par] = (double*) ddr_alloc(k * d * sizeof (double));
        sse[par] = (float*) ddr_alloc(k * sizeof (float));
        memset(len[par], 0 , k * sizeof (double));   
        memset(sum[par], 0 , k * d * sizeof (double));   
        memset(lenE[par], 0 , k * sizeof (double));   
        memset(sumE[par], 0 , k * d * sizeof (double));      
        memset(sse[par], 0 , k * sizeof (float));     
    }

    double **len_tmp = (double **) ddr_alloc(threads * sizeof (double *)); 
    double **sum_tmp = (double **) ddr_alloc(threads * sizeof (double *));
    for (int par = 0; par < threads; par++) {
        len_tmp[par] = (double*) ddr_alloc(k * sizeof (double)); 
        sum_tmp[par] = (double*) ddr_alloc(k * d * sizeof (double));
        memset(len_tmp[par], 0 , k * sizeof (double));   
        memset(sum_tmp[par], 0 , k * d * sizeof (double));      
    }    

    float *p = (float*) ddr_alloc(k * sizeof (float));
    float *l = (float*) ddr_alloc(1l * n * k * sizeof (float));   
    int *a_old = (int*) ddr_alloc(n * sizeof (int));  
    int *a = (int*) ddr_alloc(n * sizeof (int));

    float *self_d = (float*) ddr_alloc(n * sizeof (float));
    float *means_old = (float*) ddr_alloc(k * d * sizeof (float));   
    float *sigmaC = (float*) ddr_alloc(k * sizeof (float));
    memset(p, 0 , k * sizeof (float));

    int *M0 = (int*) ddr_alloc(sizeof (int));
    int *M1 = (int*) ddr_alloc(sizeof (int));
    int *data_amount = (int*) ddr_alloc(sizeof (int));
    float *accu_runtime = (float*) ddr_alloc(sizeof (float));
    *M0 = 0;
    *M1 = batchsize;
    *data_amount = 0;
    *accu_runtime = 0;

    int *epoch_count = (int*) ddr_alloc(sizeof (int));
    *epoch_count = 0;

    int iteration = 1000;
    float rho = 10;

    printf("iteration: %d; n: %d; d: %d; k: %d; batchsize: %d; seed: %d; rho: %f; mbatchsize: %d; alpha: %f\n", \
            iteration, n, d, k, batchsize, seed, rho, mbatchsize, alpha); 

    double loss = cal_loss(n, d, k, fp, dmatrix, means, threads);
    printf("initial loss: %lf\n", loss);
    fprintf(fp, "initial loss: %lf\n", loss);
    int ifchange = 0;

    for(int i = 0; i < iteration; i++){
        // if(*M1 < 6 * mbatchsize){
        if(*M1 < n / 2){
            run_nmbkm(n, d, k, 1, fp, dmatrix, means, batchsize, seed, threads,
                len_tmp, sum_tmp, sse, p, l, a_old, a, self_d, means_old, sigmaC, 
                M0, M1, data_amount, accu_runtime, rho, i);
        }
        else{
            if(ifchange == 0){
                printf("change!\n");
                fprintf(fp, "change!\n");
                ifchange = 1;
                *M1 = n;
            }
            // run_srnmbkm(n, d, k, 1, fp, dmatrix, means, batchsize, seed, threads,
            //     len, lenE, sum, sumE, sse, p, l, a_old, a, self_d, means_old, sigmaC, 
            //     M0, M1, data_amount, accu_runtime, rho, mbatchsize, epoch_count,
            //     alpha, i);
            run_srmbkm(n, d, k, 1, fp, dmatrix, means, batchsize, seed, alpha, threads, *epoch_count,
                len, lenE, sum, sumE, accu_runtime);
            *epoch_count++;
        }
        
        if((i < 9) || (i % 10 == 9)){
            loss = cal_loss(n, d, k, fp, dmatrix, means, threads);
            fprintf(fp,"iteration:%d, data:%d, time:%f, loss: %lf\n", \
            i + 1, *data_amount, *accu_runtime, loss);  
            printf("iteration:%d, data:%d, time:%f, loss: %lf\n",  \
            i + 1, *data_amount, *accu_runtime, loss);
        }
    }
    fclose(fp);
    for (int par = 0; par < threads; par++) {
        free(len[par]);   
        free(sum[par]);   
        free(sse[par]);        
    }
    free(len);  
    free(sum);  
    free(p);
    free(l);
    free(a_old);
    free(a);
    free(self_d);    
    free(means_old);
    free(sigmaC);
    free(M0);
    free(M1);
    free(data_amount);
    free(accu_runtime);
    for (int par = 0; par < threads; par++) {
        free(len_tmp[par]);   
        free(sum_tmp[par]);    
    }
    free(len_tmp);  
    free(sum_tmp);
}

int main(int argc, char** argv) {
    float *means = NULL;
    float *dmatrix = NULL;
    float *input = NULL;
    
    dsinfo dsinfo_arr[DSINFO_NUM] = {
        // {"sift", "/home/zxy/dataset/fvecs/sift/sift_base.fvecs", "fvecs", 1000000, 128},
        {"gist", "/home/zxy/dataset/fvecs/gist/gist_base.fvecs", "fvecs", 1000000, 960},
        {"poker", "/home/zxy/dataset/libsvm/poker.t", "libsvm", 1000000, 10},
        {"mnist8m", "/home/zxy/dataset/libsvm/mnist8m", "libsvm", 8100000, 784}
        };

    int k_arr[K_NUM] = {128};
    // int seed_arr[SEED_NUM] = {1, 10, 100};
    int seed_arr[SEED_NUM] = {1};
    int batchsize_arr[BS_NUM] = {1024};
    // int mbatchsize_arr[MBS_NUM] = {4, 2};
    int mbatchsize_arr[MBS_NUM] = {1};
    // float alpha_arr[A_NUM] = {0, 0.01, 0.1, 1};
    float alpha_arr[A_NUM] = {0.01};
    // float alpha_arr[A_NUM] = {0.01, 0,0.1, 1, 10, 100};
    // int mbatchsize_arr[MBS_NUM] = {1,4,8,16,1};
    int batchsize_max = 65536;

    int threads = 16;
    

    pthread_barrier_init(&barrier, NULL, threads + 1);

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
                    // initial_centroids(k, d, n, means, dmatrix);
                    // start_nmbkm(n, d, k, dsinfo_arr[dsinfo_idx].filename, dmatrix, means, batchsize, seed, threads); //打印：前10个每个都打印，前10-50个每10个打印一次

                    for(int mbs_idx = 0; mbs_idx < MBS_NUM; mbs_idx++){
                        int mbatchsize = batchsize / mbatchsize_arr[mbs_idx];

                        for(int a_idx = 0; a_idx < A_NUM; a_idx++){
                            float alpha = alpha_arr[a_idx];
                            initial_centroids(k, d, n, means, dmatrix);
                            start_srnmbkm(n, d, k, dsinfo_arr[dsinfo_idx].filename, dmatrix, means, batchsize, seed, alpha, threads, mbatchsize); //打印：只打印前10个
                        }
                    }
                }
                free(means);
            }
        }
        free(dmatrix);
        free(input);
    }
}
