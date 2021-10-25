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

#define DSINFO_NUM 1
#define K_NUM 1
#define BS_NUM 4
#define SEED_NUM 1
#define A_NUM 6

const int para_d = 16;
const int para_d_db = 8;
const int threads_loss = 16;

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
pthread_barrier_t barrier_loss;

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
    float *loss;
    double *loss_d;
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
    double *loss = ptinfo_par.loss_d;

    for (int h = 0; h < iteration; h++){
        pthread_barrier_wait(&barrier_loss);

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

        pthread_barrier_wait(&barrier_loss);
    }
    pthread_exit(NULL);
}

void *par_task_km(void *info){
    // pthread_detach(pthread_self());
    ptinfo ptinfo_par = *(ptinfo *)info;
    int batchsize_pt = ptinfo_par.batchsize_pt;
    int k = ptinfo_par.k;
    int d = ptinfo_par.d;
    double *len = ptinfo_par.len;
    double *sum = ptinfo_par.sum;
    float *dmatrix = ptinfo_par.dmatrix;
    float *means = ptinfo_par.means;
    int par = ptinfo_par.par;
    int iteration = ptinfo_par.iteration;
    float *loss = ptinfo_par.loss;

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

            //*loss += min_dist;
            
            len[index]++;
            for (int j = 0; j < d / para_d_db; j++){
                long long int tmp_p_bias = 1l * par * batchsize_pt * d + 1l * i * d + j * para_d_db;
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
                __m512d new_s_para_d = _mm512_add_pd(p_para_d, s_para_d);   
                _mm512_store_pd((sum + tmp_s_bias), new_s_para_d);                          
            }
        }  
//------------------------------------------
        // for (int i = 0; i < batchsize_pt; i++) {
        //     float min_dist = FLT_MAX;
        //     int index = 0;
        //     for (int j = 0; j < k; j++) {
        //         float dist = 0;
        //         for (int m = 0; m < d; m++){
        //             float coor_1, coor_2;
        //             //printf("par: %d, dmatrix[%d]\n", par, 1l * par * batchsize_pt * d + 1l * i * d + m);
        //             coor_1 = dmatrix[1l * par * batchsize_pt * d + 1l * i * d + m];
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
        //     for (int j = 0; j < d; j++){
        //         sum[index * d + j] += dmatrix[1l * par * batchsize_pt * d + 1l * i * d + j];
        //     }
        // }  
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

void *par_task_mbkm(void *info){
    // pthread_detach(pthread_self());
    ptinfo ptinfo_par = *(ptinfo *)info;
    int batchsize_pt = ptinfo_par.batchsize_pt;
    int n = ptinfo_par.n;
    int k = ptinfo_par.k;
    int d = ptinfo_par.d;
    double *len = ptinfo_par.len;
    double *sum = ptinfo_par.sum;
    float *dmatrix = ptinfo_par.dmatrix;
    float *means = ptinfo_par.means;
    int *rindex = ptinfo_par.rindex;
    int par = ptinfo_par.par;
    int iteration = ptinfo_par.iteration;
    int batch_num = ptinfo_par.batch_num;
    int batchsize = ptinfo_par.batchsize;
    // float *loss = ptinfo_par.loss;

    for (int h = 0; h < iteration; h++){
        pthread_barrier_wait(&barrier);

        // int bindex = h % batch_num;

        for (int i = 0; i < batchsize_pt; i++) {

            float min_dist = FLT_MAX;
            int index = 0;
            for (int j = 0; j < k; j++) {
                __m512 sq_para_d = _mm512_set1_ps(0);
                for (int m = 0; m < d / para_d; m++){
                    long long int tmp_p_bias = 1l * rindex[par * batchsize_pt + i] * d + m * para_d;
                    // long long int tmp_p_bias = 1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + m * para_d;
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
            for (int j = 0; j < d / para_d_db; j++){
                long long int tmp_p_bias = 1l * rindex[par * batchsize_pt + i] * d + j * para_d_db;
                // long long int tmp_p_bias = 1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + j * para_d_db;
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
                __m512d new_s_para_d = _mm512_add_pd(p_para_d, s_para_d);   
                _mm512_store_pd((sum + tmp_s_bias), new_s_para_d);                          
            }
        } 
//------------------------------------------        
        // for (int i = 0; i < batchsize_pt; i++) {
        //     float min_dist = FLT_MAX;
        //     int index = 0;
        //     for (int j = 0; j < k; j++) {
        //         float dist = 0;
        //         for (int m = 0; m < d; m++){
        //             float coor_1, coor_2;
        //             coor_1 = dmatrix[1l * rindex[par * batchsize_pt + i] * d + m];
        //             // coor_1 = dmatrix[1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + m];
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
        //     for (int j = 0; j < d; j++){
        //         sum[index * d + j] += dmatrix[1l * rindex[par * batchsize_pt + i] * d + j];;
        //         // sum[index * d + j] += dmatrix[1l * bindex * batchsize * d + 1l * par * batchsize_pt * d + 1l * i * d + j];

        //     }
        // }  
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
        ptinfo_arr[par].loss_d = loss + par;
		pthread_create(&Thread[par], NULL, par_task_loss, ptinfo_arr + par);
	} 
    
    pthread_barrier_wait(&barrier_loss);
    pthread_barrier_wait(&barrier_loss);
    
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

void run_km(int n, int d, int k, int epoch_num, FILE *fp, float *dmatrix, float *means,
            int batchsize, int threads,
            double **len, double **sum, double *accu_runtime){ 

    pthread_t Thread[NUM_THREADS];
    ptinfo ptinfo_arr[NUM_THREADS];
    int batchsize_pt = batchsize / threads;
	for(int par = 0; par < threads; par++){
		ptinfo_arr[par].len = len[par];
        ptinfo_arr[par].sum = sum[par];
        ptinfo_arr[par].dmatrix = dmatrix;
        ptinfo_arr[par].means = means;
        ptinfo_arr[par].batchsize_pt = batchsize_pt;
        ptinfo_arr[par].d = d;
        ptinfo_arr[par].k = k;
        ptinfo_arr[par].par = par;
        ptinfo_arr[par].iteration = epoch_num;
		pthread_create(&Thread[par], NULL, par_task_km, ptinfo_arr + par);
	} 

    CUtilTimer timer;

    for (int h = 0; h < epoch_num; h++){
        timer.start();
    
        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);

        for (int i = 0; i < k; i++) {
            int tmp_len = 0;
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

        for (int par = 0; par < threads; par++) {
            memset(len[par], 0 , k * sizeof (double));
            memset(sum[par], 0 , k * d * sizeof (double));
        }           
        timer.stop();
        *accu_runtime = *accu_runtime + timer.get_time(); 
    }

    for(int par = 0; par < threads; par++){
		pthread_join(Thread[par], NULL);
	} 
}

void run_mbkm(int n, int d, int k, int epoch_num, FILE *fp, float *dmatrix, float *means, 
                int batchsize, int seed, int threads,
                double **len, double **sum, int *random_index, double *accu_runtime){

    int batch_num = n / batchsize;

    pthread_t Thread[NUM_THREADS];
    ptinfo ptinfo_arr[NUM_THREADS];
    int batchsize_pt = batchsize / threads;
	for(int par = 0; par < threads; par++){
		ptinfo_arr[par].len = len[par];
        ptinfo_arr[par].sum = sum[par];
        ptinfo_arr[par].dmatrix = dmatrix;
        ptinfo_arr[par].means = means;
        ptinfo_arr[par].rindex = random_index;
        ptinfo_arr[par].batchsize = batchsize;
        ptinfo_arr[par].batchsize_pt = batchsize_pt;
        ptinfo_arr[par].n = n;
        ptinfo_arr[par].d = d;
        ptinfo_arr[par].k = k;
        ptinfo_arr[par].par = par;
        ptinfo_arr[par].iteration = epoch_num * batch_num;
        ptinfo_arr[par].batch_num = batch_num;
		pthread_create(&Thread[par], NULL, par_task_mbkm, &ptinfo_arr[par]);
	}
    CUtilTimer timer;
    // srand(seed);
    int epoch_count = 0;
    int printf_flag = 0;
    for (int h = 0; h < epoch_num * batch_num; h++){
        timer.start();
        for (int i = 0; i < batchsize; i++) {
            random_index[i] = rand() % n;
        }
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
    }
    for(int par = 0; par < threads; par++){
		pthread_join(Thread[par], NULL);
	} 
}

void run_srmbkm(int n, int d, int k, int epoch_num, FILE *fp, float *dmatrix, float *means, \
            int batchsize, int seed, double alpha, int threads, int epoch_count,
            double **len, double **lenE, double **sum, double **sumE, double *accu_runtime){

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

void start_km(int n, int d, int k, char *filename, float *dmatrix, float *means, int seed, int threads){
    char fout[100];
    sprintf(fout, "%s/km/%s_k%d_sd%d.txt", fout_root, filename, k, seed);
    FILE *fp;
    if((fp=fopen(fout,"wt+")) == NULL){
        printf("Cannot open %s!", fout);
        exit(1);
    }
    int batchsize = n;
    double **len = NULL;
    double **sum = NULL;
    len = (double **) ddr_alloc(threads * sizeof (double *)); 
    sum = (double **) ddr_alloc(threads * sizeof (double *));
    for (int par = 0; par < threads; par++) {
        len[par] = (double*) ddr_alloc(k * sizeof (double)); 
        sum[par] = (double*) ddr_alloc(k * d * sizeof (double));
        memset(len[par], 0 , k * sizeof (double));   
        memset(sum[par], 0 , k * d * sizeof (double));           
    }
    double *accu_runtime = (double*) ddr_alloc(sizeof (double));
    accu_runtime[0] = 0.0;

    // 批样本大小batch size,仅用所有数据能容纳的batch size个样本来做K-Means聚类，其他数据丢
    int epoch_num = 300;  
    printf("epoch_num: %d; data size: %d; dimension: %d; number of cluster: %d; batchsize:%d\n", epoch_num, n, d, k, batchsize);
    double loss = cal_loss(n, d, k, fp, dmatrix, means, threads_loss);
    fprintf(fp, "initial loss: %lf\n", loss);
    printf("initial loss: %lf\n", loss);
    for(int i = 0; i < epoch_num; i++){
        // if((i < 9) || (i % 10 == 9 && i < 100) || (i % 100 == 99 && i < 1000) || (i % 1000 == 999)){
        run_km(n, d, k, 1, fp, dmatrix, means, batchsize, threads,
                    len, sum, accu_runtime);
        if((i < 9) || (i % 10 == 9)){
            loss = cal_loss(n, d, k, fp, dmatrix, means, threads_loss);
            fprintf(fp,"iteration:%d, epoch:%d, data:%d, time:%f, loss: %lf\n", \
            (i + 1) * n / batchsize, i + 1, (i + 1) * n, *accu_runtime, loss);  
            printf("iteration:%d, epoch:%d, data:%d, time:%f, loss: %lf\n",  \
            (i + 1) * n / batchsize, i + 1, (i + 1) * n, *accu_runtime, loss);
        }
    }
    fclose(fp);
    for (int par = 0; par < threads; par++) {
        free(len[par]);   
        free(sum[par]);           
    }
    free(len);
    free(sum);
}

void start_mbkm(int n, int d, int k, char *filename, float *dmatrix, float *means, int batchsize, int seed, int threads){
    char fout[100];
    sprintf(fout, "%s/mbkm/%s_k%d_sd%d_bs%d.txt", \
            fout_root, filename, k, seed, batchsize);
    FILE *fp;
    if((fp=fopen(fout,"wt+")) == NULL){
        printf("Cannot open %s!", fout);
        exit(1);
    }
    double **len = NULL;
    double **sum = NULL;
    len = (double **) ddr_alloc(threads * sizeof (double *)); 
    sum = (double **) ddr_alloc(threads * sizeof (double *));
    for (int par = 0; par < threads; par++) {
        len[par] = (double*) ddr_alloc(k * sizeof (double)); 
        sum[par] = (double*) ddr_alloc(k * d * sizeof (double));
        memset(len[par], 0 , k * sizeof (int));   
        memset(sum[par], 0 , k * d * sizeof (double));           
    }
    int *random_index = NULL;
    random_index = (int*) ddr_alloc(batchsize * sizeof (int));
    double *accu_runtime = (double*) ddr_alloc(sizeof (double));
    accu_runtime[0] = 0.0;
    int epoch_num = 50;
    srand(seed);
    printf("epoch_num: %d; n: %d; d: %d; k: %d; batchsize: %d; seed: %d\n", \
            epoch_num, n, d, k, batchsize, seed); 
    double loss = cal_loss(n, d, k, fp, dmatrix, means, threads_loss);
    printf("initial loss: %lf\n", loss);
    fprintf(fp, "initial loss: %lf\n", loss);
    for(int i = 0; i < epoch_num; i++){
        run_mbkm(n, d, k, 1, fp, dmatrix, means, batchsize, seed, threads,
                len, sum, random_index, accu_runtime);
        if((i < 9) || (i % 10 == 9)){
            loss = cal_loss(n, d, k, fp, dmatrix, means, threads_loss);
            fprintf(fp,"iteration:%d, epoch:%d, data:%d, time:%f, loss: %lf\n", \
            (i + 1) * n / batchsize, i + 1, (i + 1) * n, *accu_runtime, loss);  
            printf("iteration:%d, epoch:%d, data:%d, time:%f, loss: %lf\n",  \
            (i + 1) * n / batchsize, i + 1, (i + 1) * n, *accu_runtime, loss);
        }
    }
    fclose(fp);

    free(random_index);
    for (int par = 0; par < threads; par++) {
        free(len[par]);   
        free(sum[par]);           
    }
    free(len);
    free(sum);
    free(accu_runtime);
}

void start_srmbkm(int n, int d, int k, char *filename, float *dmatrix, float *means, int batchsize, int seed, double alpha, int threads){
    char fout[100];
    sprintf(fout, "%s/srmbkm/%s_k%d_sd%d_bs%d_a%.3lf.txt", \
            fout_root, filename, k, seed, batchsize, alpha);
    FILE *fp;
    if((fp=fopen(fout,"wt+")) == NULL){
        printf("Cannot open %s!", fout);
        exit(1);
    }
    double **len = NULL;
    double **sum = NULL;
    double **lenE = NULL;
    double **sumE = NULL;
    len = (double **) ddr_alloc(threads * sizeof (double *)); 
    lenE = (double **) ddr_alloc(threads * sizeof (double *)); 
    sum = (double **) ddr_alloc(threads * sizeof (double *));
    sumE = (double **) ddr_alloc(threads * sizeof (double *));
    for (int par = 0; par < threads; par++) {
        len[par] = (double*) ddr_alloc(k * sizeof (double)); 
        lenE[par] = (double*) ddr_alloc(k * sizeof (double)); 
        sum[par] = (double*) ddr_alloc(k * d * sizeof (double));
        sumE[par] = (double*) ddr_alloc(k * d * sizeof (double)); 
        memset(len[par], 0 , k * sizeof (double));  
        memset(lenE[par], 0 , k * sizeof (double));  
        memset(sum[par], 0 , k * d * sizeof (double));  
        memset(sumE[par], 0 , k * d * sizeof (double));              
    } 
    double *accu_runtime = (double*) ddr_alloc(sizeof (double));
    accu_runtime[0] = 0.0;
    int epoch_num = 200;

    printf("epoch_num: %d; n: %d; d: %d; k: %d; batchsize: %d; seed: %d; alpha: %lf\n", \
        epoch_num, n, d, k, batchsize, seed, alpha); 
    double loss = cal_loss(n, d, k, fp, dmatrix, means, threads_loss);
    fprintf(fp, "initial loss: %lf\n", loss);
    printf("initial loss: %lf\n", loss);
    for(int i = 0; i < epoch_num; i++){
        run_srmbkm(n, d, k, 1, fp, dmatrix, means, batchsize, seed, alpha, threads, i,
                len, lenE, sum, sumE, accu_runtime);
        if((i < 9) || (i % 10 == 9)){
            loss = cal_loss(n, d, k, fp, dmatrix, means, threads_loss);
            fprintf(fp,"iteration:%d, epoch:%d, data:%d, time:%f, loss: %lf\n", \
            (i + 1) * n / batchsize, i + 1, (i + 1) * n, *accu_runtime, loss);  
            printf("iteration:%d, epoch:%d, data:%d, time:%f, loss: %lf\n",  \
            (i + 1) * n / batchsize, i + 1, (i + 1) * n, *accu_runtime, loss);
        }
    }

    fclose(fp);

    for (int par = 0; par < threads; par++) {
        free(len[par]);   
        free(lenE[par]);  
        free(sum[par]);     
        free(sumE[par]);      
    }
    free(len);
    free(lenE);
    free(sum);
    free(sumE);
    free(accu_runtime);
}


int main(int argc, char** argv) {
    float *means = NULL;
    float *dmatrix = NULL;
    float *input = NULL;
    
    dsinfo dsinfo_arr[DSINFO_NUM] = {
        // {"sift", "/home/zxy/dataset/fvecs/sift/sift_base.fvecs", "fvecs", 1000000, 128}
        // {"poker", "/home/zxy/dataset/libsvm/poker.t", "libsvm", 1000000, 10},
        // {"sift", "/home/zxy/dataset/fvecs/sift/sift_base.fvecs", "fvecs", 1000000, 128},
        // {"poker", "/home/zxy/dataset/libsvm/poker.t", "libsvm", 1000000, 10},
        // {"gist", "/home/zxy/dataset/fvecs/gist/gist_base.fvecs", "fvecs", 1000000, 960}
        {"mnist8m", "/home/zxy/dataset/libsvm/mnist8m", "libsvm", 8100000, 784},
        };

    int k_arr[K_NUM] = {128};
    // int k_arr[K_NUM] = {16};
    int seed_arr[SEED_NUM] = {1};
    int batchsize_arr[BS_NUM] = {1024, 4096, 16384, 65536};
    // int batchsize_arr[BS_NUM] = {4096};
    float alpha_arr[A_NUM] = {0, 0.01, 0.1, 1, 10, 100};
    int batchsize_max = 65536;
    // int batchsize_arr[BS_NUM] = {983040, 4096, 16384, 65536};
    int threads = 16;
    
    

    pthread_barrier_init(&barrier, NULL, threads + 1);
    pthread_barrier_init(&barrier_loss, NULL, threads_loss + 1);

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
                initial_centroids(k, d, n, means, dmatrix);
                start_km(n, d, k, dsinfo_arr[dsinfo_idx].filename, dmatrix, means, seed, threads); //打印：前10个每个都打印，前10-50个每10个打印一次

                // for(int bs_idx = 0; bs_idx < 1; bs_idx++){
                //     int batchsize = batchsize_arr[bs_idx];
                //     // int batchsize = batchsize_arr[k_idx];
                //     // initial_centroids(k, d, n, means, dmatrix);
                //     // start_mbkm(n, d, k, dsinfo_arr[dsinfo_idx].filename, dmatrix, means, batchsize, seed, threads); //打印：前10个每个都打印，前10-50个每10个打印一次

                //     for(int a_idx = 0; a_idx < A_NUM; a_idx++){
                //         double alpha = alpha_arr[a_idx];
                //         initial_centroids(k, d, n, means, dmatrix);
                //         start_srmbkm(n, d, k, dsinfo_arr[dsinfo_idx].filename, dmatrix, means, batchsize, seed, alpha, threads); //打印：只打印前10个
                //     }
                // }
                free(means);
            }
        }
        free(dmatrix);
        free(input);
    }
}
