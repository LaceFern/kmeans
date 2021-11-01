#include <vector>
#include <stdint.h>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
	
// #include "../util/allocation.h"
// #include "../util/dataIo.h"
// #include "../util/arguments.h"
// #include "../mckm/mckm.h"
// #include "../util/timer.h"
// #include "../cmdparser/cmdlineparser.h"

#include "allocation.h"
#include "dataIo.h"
#include "arguments.h"
#include "mckm.h"
#include "timer.h"
#include "cmdlineparser.h"
	

#define DSINFO_NUM 4
#define K_NUM 3
#define BS_NUM 4
#define MBS_NUM 5
#define SEED_NUM 3
#define A_NUM 6

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

char *fout_root = "/home/zxy/final/next_results_2/";

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
    float *loss;
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
    float *loss = ptinfo_par.loss;

    for (int h = 0; h < iteration; h++){
        pthread_barrier_wait(&barrier);

        // for (int i = 0; i < batchsize_pt; i++) {
        //     float min_dist = INFINITY;
        //     int index = 0;
        //     for (int j = 0; j < k; j++) {

        //         __m512 sq_para_d = _mm512_set1_ps(0);
        //         for (int m = 0; m < d / para_d; m++){
        //             long long int tmp_p_bias = 1l * par * batchsize_pt * d + 1l * i * d + m * para_d;
        //             int tmp_c_bias = j * d + m * para_d;
        //             __m512 p_para_d = _mm512_load_ps((dmatrix + tmp_p_bias));
        //             __m512 c_para_d = _mm512_load_ps((means + tmp_c_bias)); 
        //             __m512 sub_para_d = _mm512_sub_ps(p_para_d, c_para_d);
        //             sq_para_d = _mm512_fmadd_ps(sub_para_d, sub_para_d, sq_para_d);
        //         }
        //         float dist = _mm512_reduce_add_ps(sq_para_d);

        //         if (dist < min_dist) { 
        //             min_dist = dist;
        //             index = j;
        //         }
        //     }
        //     *loss += min_dist;
        // }  
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

float cal_loss(int n, int d, int k, FILE *fp, float *dmatrix, float *means, int threads){

    float *loss = (float*) ddr_alloc(threads * sizeof (float)); 
    memset(loss, 0 , threads * sizeof (float)); 

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
    
    float loss_total = 0;
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
                double *len, double *sum, float *sse, float *p, float *l, 
                int *a_old, int *a, float *self_d, float *means_old, float *sigmaC, 
                int *M0, int *M1, int *data_amount, float *accu_runtime, float rho){

    CUtilTimer timer;
    for (int h = 0; h < iteration; h++){
        timer.start();
        for (int i = 0; i < *M0; i++){
            for (int j = 0; j < k; j++){
                l[i * k + j] -= p[j];
            }
        }
        for (int i = 0; i < *M0; i++){
            a_old[i] = a[i];
            sse[a_old[i]] -= self_d[i] * self_d[i];                                                                       
            for (int j = 0; j < d; j++){
                sum[a_old[i] * d + j] -= dmatrix[i * d + j];
            }
            len[a_old[i]] -= 1;
            //assignment_with_bounds(i)
            self_d[i] = 0;
            for (int j = 0; j < d; j++){
                self_d[i] += (means[a[i] * d + j] - dmatrix[i * d + j]) * (means[a[i] * d + j] - dmatrix[i * d + j]); 
            }
            self_d[i] = sqrt(self_d[i]);
            for (int j = 0; j < k; j++){
                if (j != a[i]){
                    if (l[i * k + j] < self_d[i]){
                        l[i * k + j] = 0;
                        for (int kk = 0; kk < d; kk++){
                            l[i * k + j] += (means[j * d + kk] - dmatrix[i * d + kk]) * (means[j * d + kk] - dmatrix[i * d + kk]); 
                        }
                        l[i * k + j] = sqrt(l[i * k + j]);
                        if (l[i * k + j] < self_d[i]){
                            a[i] = j;
                            self_d[i] = l[i * k + j];
                        }
                    }
                }
            }
            //accumulate(i)
            for (int j = 0; j < d; j++){
                sum[a[i] * d + j] += dmatrix[i * d + j];
            }
            len[a[i]] += 1;
            //
            sse[a[i]] += self_d[i] * self_d[i];
            //printf("checkpoint0 a[i]:%d, len:%d\n", a[i], len[a[i]]);
        }    

        for (int i = *M0; i < *M1; i++){
            for (int j = 0; j < k; j++){
                l[i * k + j] = 0;
                for (int kk = 0; kk < d; kk++){
                    l[i * k + j] += (means[j * d + kk] - dmatrix[i * d + kk]) * (means[j * d + kk] - dmatrix[i * d + kk]); 
                }
                l[i * k + j] = sqrt(l[i * k + j]);
            }
            int min_index = 0;
            float min_l = INFINITY;
            for (int j = 0; j < k; j++){
                if (l[i * k + j] < min_l){
                    min_l = l[i * k + j];
                    min_index = j;
                }
            }
            //std::cout<<"min_index="<<min_index<<"\n";
            a[i] = min_index;
            self_d[i] = l[i * k + a[i]];
            for (int j = 0; j < d; j++){
                sum[a[i] * d + j] += dmatrix[i * d + j];
            }
            len[a[i]] += 1;  
            //printf("a[i]:%d, len:%d\n", a[i], len[a[i]]);      
            sse[a[i]] += self_d[i] * self_d[i];
        }

        for (int j = 0; j < k; j++){
            sigmaC[j] = sqrt(sse[j] / (len[j] * (len[j] - 1)));
            //printf("sse:%lf, len:%d\n", sse[j], len[j]);
            p[j] = 0;
            for (int kk = 0; kk < d; kk++){
                means_old[j * d + kk] = means[j * d + kk];
                if (len[j] > 0){
                    means[j * d + kk] = sum[j * d + kk] / len[j];
                } 
                p[j] += (means[j * d + kk] - means_old[j * d + kk]) * (means[j * d + kk] - means_old[j * d + kk]);
            }           
            p[j] = sqrt(p[j]);
        }
        
        float min_rho = INFINITY;
        for (int j = 0; j < k; j++){
            float tmp_rho = sigmaC[j] / p[j];
            //printf("sigmaC:%f, p:%f\n", sigmaC[j], p[j]);
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
            *M0 = *M1;//????
        }
        timer.stop();
        *accu_runtime += timer.get_time();
        *data_amount += *M0;
    }
}

void run_srnmbkm(int n, int d, int k, int iteration, FILE *fp, float *dmatrix, float *means, 
                int batchsize, int seed, int threads,
                double *len, double *lenE, double *sum, double *sumE, float *sse, float *p, float *l, 
                int *a_old, int *a, float *self_d, float *means_old, float *sigmaC, 
                int *M0, int *M1, int *data_amount, float *accu_runtime, float rho, int mbatchsize, int *epoch_count,
                float alpha){
    CUtilTimer timer;
    for (int h = 0; h < iteration; h++){
        
        timer.start();
        for (int j = 0; j < k; j++){
            for (int kk = 0; kk < d; kk++){
                means_old[j * d + kk] = means[j * d + kk];
            }           
        }
        
        memset(sse, 0 , k * sizeof (float)); 
        for (int m = 0; m < *M1 / mbatchsize; m++) {
            for (int i = 0; i < mbatchsize; i++) {
                float min_dist = INFINITY;
                int index = 0;
                for (int j = 0; j < k; j++) {
                    float dist = 0.0f;
                    for (int kk = 0; kk < d; kk++){
                        float coor_1, coor_2;
                        coor_1 = dmatrix[m * mbatchsize * d + i * d + kk];
                        coor_2 = means[j * d + kk];
                        dist += (coor_1 - coor_2)*(coor_1 - coor_2);
                        //dist += 0.5*coor_2*coor_2 - coor_1*coor_2;
                    
                    }
                    if (dist < min_dist) { /* find the min and its array index */
                        min_dist = dist;
                        index = j;
                    }
                }
                //std::cout<<"index = "<<index<<"\n";

                len[index]++;
                lenE[index]++;
                for (int j = 0; j < d; j++){
                    sum[index * d + j] += dmatrix[m * mbatchsize * d + i * d + j];
                    sumE[index * d + j] += dmatrix[m * mbatchsize * d + i * d + j];
                }
                sse[index] += min_dist * min_dist;
            }  
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < d; j++) {
                    if (len[i] > 0){
                        means[i * d + j] = sum[i * d + j] / len[i];
                    }
                }
            }
        }

        for (int j = 0; j < k; j++){
            sigmaC[j] = sqrt(sse[j] / (lenE[j] * (lenE[j] - 1)));
            p[j] = 0;
            for (int kk = 0; kk < d; kk++){
                p[j] += (means[j * d + kk] - means_old[j * d + kk]) * (means[j * d + kk] - means_old[j * d + kk]);
            }           
            p[j] = sqrt(p[j]);
        }

        float min_rho = INFINITY;
        for (int j = 0; j < k; j++){
            float tmp_rho = sigmaC[j] / p[j];
            //printf("sigmaC:%f, p:%f\n", sigmaC[j], p[j]);
            if (!isnan(tmp_rho) && tmp_rho < min_rho){
                min_rho = tmp_rho;
            }
        }
        int M_old = *M1;
        if (min_rho > rho && *M1 != n){
	
            *M1 = (2 * *M1) < n ? (2 * *M1) : n;
            *epoch_count = 0;
            memset(len, 0 , k * sizeof (double)); 
            memset(lenE, 0 , k * sizeof (double));  
            memset(sum, 0 , k * d * sizeof (double)); 
            memset(sumE, 0 , k * d * sizeof (double)); 
        }
        else{
            (*epoch_count)++;
            double coefficient = *epoch_count * alpha;
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < d; j++) {
                    if (lenE[i] >= 1){
                        means[i * d + j] = sumE[i * d + j] / lenE[i];
                    }
                }
            } 
            for (int i = 0; i < k; i++) {
                len[i] = lenE[i] * coefficient;
                for (int j = 0; j < d; j++) {
                    sum[i * d + j] = sumE[i * d + j] * coefficient;
                }
            }
	 
            memset(lenE, 0 , k * sizeof (double)); 
            memset(sumE, 0 , k * d * sizeof (double));

        }
        //if(h % batch_num == batch_num - 1){
        timer.stop();
        *accu_runtime += timer.get_time();
        *data_amount += M_old;
        //}
    }
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

    double *len = NULL;
    double *sum = NULL;
    len = (double*) ddr_alloc(k * sizeof (double)); 
    sum = (double*) ddr_alloc(k * d * sizeof (double));
    memset(len, 0 , k * sizeof (double));  
    memset(sum, 0 , k * sizeof (double));  

    float *sse = (float*) ddr_alloc(k * sizeof (float));
    float *p = (float*) ddr_alloc(k * sizeof (float));
    float *l = (float*) ddr_alloc(n * k * sizeof (float));   
    int *a_old = (int*) ddr_alloc(n * sizeof (int));  
    int *a = (int*) ddr_alloc(n * sizeof (int));

    float *self_d = (float*) ddr_alloc(n * sizeof (float));
    float *means_old = (float*) ddr_alloc(k * d * sizeof (float));   
    float *sigmaC = (float*) ddr_alloc(k * sizeof (float));
    memset(sse, 0 , k * sizeof (float)); 
    memset(p, 0 , k * sizeof (float));

    int *M0 = (int*) ddr_alloc(sizeof (int));
    int *M1 = (int*) ddr_alloc(sizeof (int));
    int *data_amount = (int*) ddr_alloc(sizeof (int));
    float *accu_runtime = (float*) ddr_alloc(sizeof (float));
    *M0 = 0;
    *M1 = batchsize;
    *data_amount = 0;
    *accu_runtime = 0;

    //printf("data:%d, loss:%f, time:%f\n", *data_amount, loss, *accu_runtime);

    int iteration = 5000;
    float rho = 10;

    printf("iteration: %d; n: %d; d: %d; k: %d; batchsize: %d; seed: %d; rho: %f\n", \
            iteration, n, d, k, batchsize, seed, rho); 

    float loss = cal_loss(n, d, k, fp, dmatrix, means, threads);
    fprintf(fp, "initial loss: %f\n", loss);
    for(int i = 0; i < iteration; i++){
        run_nmbkm(n, d, k, 1, fp, dmatrix, means, batchsize, seed, threads,
                len, sum, sse, p, l, a_old, a, self_d, means_old, sigmaC, 
                M0, M1, data_amount, accu_runtime, rho);
        if((i < 9) || (i % 10 == 9)){
            loss = cal_loss(n, d, k, fp, dmatrix, means, threads);
            fprintf(fp,"iteration:%d, data:%d, time:%f, loss: %f\n", \
            i + 1, *data_amount, *accu_runtime, loss);  
            printf("iteration:%d, data:%d, time:%f, loss: %f\n",  \
            i + 1, *data_amount, *accu_runtime, loss);
        }
    }
    fclose(fp);

    free(len);
    free(sum);
    free(sse);
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

void start_srnmbkm(int n, int d, int k, char *filename, float *dmatrix, float *means, int batchsize, int seed, double alpha, int threads, int mbatchsize){

    char fout[100];
    sprintf(fout, "%s/srnmbkm/%s_k%d_sd%d_bs%d.txt", \
            fout_root, filename, k, seed, batchsize);
    FILE *fp;
    if((fp=fopen(fout,"wt+")) == NULL){
        printf("Cannot open %s!", fout);
        exit(1);
    }

    double *len = NULL;
    double *sum = NULL;
    double *lenE = NULL;
    double *sumE = NULL;
    len = (double*) ddr_alloc(k * sizeof (double)); 
    lenE = (double*) ddr_alloc(k * sizeof (double)); 
    sum = (double*) ddr_alloc(k * d * sizeof (double));
    sumE = (double*) ddr_alloc(k * d * sizeof (double));
    memset(len, 0 , k * sizeof (double));  
    memset(lenE, 0 , k * sizeof (double));  
    memset(sum, 0 , k * d * sizeof (double));  
    memset(sumE, 0 , k * d * sizeof (double));   

    float *sse = (float*) ddr_alloc(k * sizeof (float));
    float *p = (float*) ddr_alloc(k * sizeof (float));
    float *l = (float*) ddr_alloc(n * k * sizeof (float));   
    int *a_old = (int*) ddr_alloc(n * sizeof (int));  
    int *a = (int*) ddr_alloc(n * sizeof (int));

    float *self_d = (float*) ddr_alloc(n * sizeof (float));
    float *means_old = (float*) ddr_alloc(k * d * sizeof (float));   
    float *sigmaC = (float*) ddr_alloc(k * sizeof (float));
    memset(sse, 0 , k * sizeof (float)); 
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

    int iteration = 5000;
    float rho = 10;

    printf("iteration: %d; n: %d; d: %d; k: %d; batchsize: %d; seed: %d; rho: %f; mbatchsize: %d; alpha: %f\n", \
            iteration, n, d, k, batchsize, seed, rho, mbatchsize, alpha); 

    float loss = cal_loss(n, d, k, fp, dmatrix, means, threads);
    fprintf(fp, "initial loss: %f\n", loss);
    for(int i = 0; i < iteration; i++){
        run_srnmbkm(n, d, k, 1, fp, dmatrix, means, batchsize, seed, threads,
                len, lenE, sum, sumE, sse, p, l, a_old, a, self_d, means_old, sigmaC, 
                M0, M1, data_amount, accu_runtime, rho, mbatchsize, epoch_count,
                alpha);
        if((i < 9) || (i % 10 == 9)){
            loss = cal_loss(n, d, k, fp, dmatrix, means, threads);
            fprintf(fp,"iteration:%d, data:%d, time:%f, loss: %f\n", \
            i + 1, *data_amount, *accu_runtime, loss);  
            printf("iteration:%d, data:%d, time:%f, loss: %f\n",  \
            i + 1, *data_amount, *accu_runtime, loss);
        }
    }
    fclose(fp);

    free(len);
    free(sum);
    free(sse);
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

int main(int argc, char** argv) {
    float *means = NULL;
    float *dmatrix = NULL;
    float *input = NULL;
    
    dsinfo dsinfo_arr[DSINFO_NUM] = {
        {"sift", "/home/zxy/dataset/fvecs/sift/sift_base.fvecs", "fvecs", 1000000, 128},
        {"gist", "/home/zxy/dataset/fvecs/gist/gist_base.fvecs", "fvecs", 1000000, 960},
        {"poker", "/home/zxy/dataset/libsvm/poker.t", "libsvm", 1000000, 10},
        {"mnist8m", "/home/zxy/dataset/libsvm/mnist8m", "libsvm", 8100000, 784}};

    int k_arr[K_NUM] = {16, 64, 256};
    int seed_arr[SEED_NUM] = {1, 10, 100};
    int batchsize_arr[BS_NUM] = {1024, 4096, 16384, 65536};
    // int mbatchsize_arr[BS_NUM] = {16, 8, 4, 2, 1};
    float alpha_arr[A_NUM] = {0, 0.01, 0.1, 1, 10, 100};
    int mbatchsize_arr[MBS_NUM] = {1, 16, 8, 4, 2};

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
                n = n / batchsize_arr[BS_NUM - 1] * batchsize_arr[BS_NUM - 1];// 重新设置n为各采样点的最大公倍数
                int k = k_arr[k_idx];
                means = (float*) ddr_alloc(1l * k * d * sizeof (float));  // means = clusters

                for(int bs_idx = 0; bs_idx < BS_NUM; bs_idx++){
                    int batchsize = batchsize_arr[bs_idx];
                    initial_centroids(k, d, n, means, dmatrix);
                    start_nmbkm(n, d, k, dsinfo_arr[dsinfo_idx].filename, dmatrix, means, batchsize, seed, threads); //打印：前10个每个都打印，前10-50个每10个打印一次

                    for(int mbs_idx = 0; mbs_idx < MBS_NUM; mbs_idx++){
                        int mbatchsize = batchsize_arr[bs_idx] / mbatchsize_arr[mbs_idx];

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