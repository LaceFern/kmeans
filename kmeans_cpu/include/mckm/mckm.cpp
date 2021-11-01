#include <iostream>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>
#include "mckm.h"
#include "../util/timer.h"


using namespace std;


void compute_reference_kmeans(float* objects, float* clusters_ref,int numObjs, int numClusters, int numCoords, int iter,unsigned int* member_ref, float threshold){

    
    printf("\n\nReference floating point kmeans start\n");
    

    int loop = 0;

    //initialize temp arrays
    unsigned int mem_size_cluster = sizeof(float)* numClusters * numCoords;
    float* newClusters = NULL;
    int status = posix_memalign((void**)&newClusters, 64, mem_size_cluster);
    
    unsigned int mem_size_cluster_size = sizeof(int)* numClusters;
    unsigned int * newClusterSize = NULL;
    status=posix_memalign((void**)&newClusterSize, 64, mem_size_cluster_size);

    float center_change;
    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    for (int i=0; i<numClusters; i++){
        newClusterSize[i] = 0;
        for (int j=0; j<numCoords; j++){
            newClusters[i*numCoords+j] = 0.0f;
        }
    }

    //float* clusters_ref = NULL;
    //status=posix_memalign((void**)&clusters_ref, 64, sizeof(float)*numClusters*numCoords);
    //memcpy(clusters_ref, initial_centroid, sizeof(float)*numClusters*numCoords);

    do {
        for (int i=0; i<numObjs; i++) {
            
            float min_dist = FLT_MAX;
            int index = 0;
        
           // printf("dist from sample %d:", i);
            for (int j=0; j<numClusters; j++) {
                float dist = 0.0f;
                for (int k=0; k<numCoords; k++){
                    float coor_1, coor_2;
                    coor_1 = objects [i*numCoords+k];
                    coor_2 = clusters_ref [j*numCoords+k];
                    //dist += (coor_1 - coor_2)*(coor_1 - coor_2);
                    dist += 0.5*coor_2*coor_2 - coor_1*coor_2;
                    
                }
                if (dist < min_dist) { /* find the min and its array index */
                    min_dist = dist;
                    index    = j;
                }
               // printf("c%d: %f ", j, dist);
            }
	   // printf("\n");
            // assign the membership to object i
            member_ref[i] = short(index);

            // upsate new cluster centers : sum of objects located within 
            newClusterSize[index]++;
            for (int j=0; j<numCoords; j++){
                           newClusters[index*numCoords+j] += objects[i*numCoords+j];
            }
        }

        center_change = 0.0f;
        // average the sum and replace old cluster centers with newClusters 
        for (int i=0; i<numClusters; i++) {
            for (int j=0; j<numCoords; j++) {
                float cluster_tmp = clusters_ref[i*numCoords+j];
                if (newClusterSize[i] > 0){
                    clusters_ref[i*numCoords+j] = newClusters[i*numCoords+j] / newClusterSize[i];
                }
                center_change += (cluster_tmp - clusters_ref[i*numCoords+j]) * (cluster_tmp - clusters_ref[i*numCoords+j]);
                newClusters[i*numCoords+j] = 0.0f;  
            }
            newClusterSize[i] = 0;  
        }
        float loss = get_sse( numObjs, numClusters, numCoords, objects, clusters_ref);
    	printf("iteration:%d, center change:%f, loss:%f\n", loop, center_change, loss);   
    } while (++loop < iter & center_change>threshold);
    printf("Final iteration:%d\n", loop);

    // printf("Reference cluster assignment\n");
    // for (int i = 0; i < numObjs; ++i)
    // {
    //    printf("%d ", member_ref[i]);
    // }

   /*  printf("\nReference cluster centroids\n");
     for (int i = 0; i < numClusters; ++i)
     {
         for (int j = 0; j < numCoords; ++j)
         {
            printf("%f ", clusters_ref[i*numCoords+j]);
         }
         printf("\n");
     }*/

    //Clean temp array
    free (newClusterSize);
    free (newClusters);
}


void compare_with_reference(unsigned int* cID, unsigned int* member_ref, float* means, float* means_ref, const int n, const int d, const int k)
{
 //    printf("compare assignment:\n");
 //    int assignment_diff_cnt = 0;
 //    for (int i = 0; i < n; ++i)
 //    {
 //        if (cID[i] != member_ref[i])
 //        {
 //            assignment_diff_cnt++;
 //          //  printf("assignment difference on sample %d, reference:%u, result:%u\n", i, member_ref[i], cID[i]);
 //        }
 //    }
 //    if (assignment_diff_cnt == 0)
 //    {
 //        printf("All assignments correct\n");
 //    }
 //    else
	// printf("%d assignments not equal!!!\n", assignment_diff_cnt);

    printf("compare centroids\n");
    int centroid_diff_cnt = 0;
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < d; ++j)
	{
	 
            if( ( means[i*d+j] - means_ref[i*d+j] > 0.1 ) | (means_ref[i*d+j] - means[i*d+j] > 0.1 ))
            {
                centroid_diff_cnt ++;
        //        printf("sample%d, dim%d not equal: reference:%f, result:%f\n",i, j, means_ref[i*d+j], means[i*d+j] );
            }
        }
    }
    if (centroid_diff_cnt == 0)
    {
        printf("All centroids correct\n");
    }
    else 
	printf("Total%d centroid dim not equal!!\n", centroid_diff_cnt);


}


float get_change_center_thres (float* features, int nfeatures, int npoints, float thre)
{
    float* mean = (float *)malloc(nfeatures*sizeof(float)); 
    float* var = (float *)malloc(nfeatures*sizeof(float)); 

    float mean_var = 0.0f;
    float threshold = 0.0f;
    for (int j = 0; j < nfeatures; ++j)
    {
        mean[j] = 0;
        float accu = 0;
        for (int i = 0; i < npoints; ++i)
        {
            accu += features[i*nfeatures+j];
        }
        mean[j] = accu/npoints;
    }
    // printf("variance:\n");
    for (int j = 0; j < nfeatures; ++j)
    {
        var[j] = 0;
        for (int i = 0; i < npoints; ++i)
        {
            var[j] += (features[i*nfeatures+j] - mean[j])*(features[i*nfeatures+j] - mean[j]);
        }
        var[j] = var[j]/npoints;
        // printf("%lf ", var[j]);
    }
    // printf("\n");

    for (int i = 0; i < nfeatures; ++i)
    {
        mean_var += var[i];
    }
    mean_var = mean_var/nfeatures;
    threshold = mean_var*thre;
    // printf("mean_var: %lf\n", mean_var);
    return threshold;

}


void shuffle_object (float * objects, int numCoords, int numObjs, float* shuffled_objects,int seed)
{
    vector<int> index;
    for (int i = 0; i < numObjs; ++i)
    {
        index.push_back(i);
    }
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (index.begin(), index.end(), std::default_random_engine(seed));

    //printf("shuffle index\n");
    for (int i = 0; i < numObjs; ++i)
    {
      //   printf("%d ", index[i]);
        for (int j = 0; j < numCoords; ++j)
        {
            shuffled_objects[1l * i * numCoords + j] = objects[1l * index[i] * numCoords + j];
        }
    }
}


void shuffle_object_int (int * objects, int numCoords, int numObjs, int* shuffled_objects,int seed)
{
    vector<int> index;
    for (int i = 0; i < numObjs; ++i)
    {
        index.push_back(i);
    }
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (index.begin(), index.end(), std::default_random_engine(seed));

    //printf("shuffle index\n");
    for (int i = 0; i < numObjs; ++i)
    {
      //   printf("%d ", index[i]);
        for (int j = 0; j < numCoords; ++j)
        {
            shuffled_objects[i*numCoords+j] = objects[index[i]*numCoords+j];
        }
    }
}


void shuffle_object_double ( double* objects, int numCoords, int numObjs, double* shuffled_objects,int seed)
{
    vector<int> index;
    for (int i = 0; i < numObjs; ++i)
    {
        index.push_back(i);
    }
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle (index.begin(), index.end(), std::default_random_engine(seed));

    //printf("shuffle index\n");
    for (int i = 0; i < numObjs; ++i)
    {
      //   printf("%d ", index[i]);
        for (int j = 0; j < numCoords; ++j)
        {
            shuffled_objects[i*numCoords+j] = objects[index[i]*numCoords+j];
        }
    }
}


float get_sse(int numObjs, int numClusters, int numCoords, float * objects, float * clusters_ref)
{
    float loss = 0.0f;
    for (int i=0; i<numObjs; i++) {
        float min_dist = INFINITY;
        int index = 0;
        for (int j=0; j<numClusters; j++) {
            float dist = 0.0f;
            for (int k=0; k<numCoords; k++){
                float coor_1, coor_2;
                coor_1 = objects [1l * i * numCoords + k];
                coor_2 = clusters_ref [1l * j*numCoords+k];
                dist += (coor_1 - coor_2)*(coor_1 - coor_2);
                
            }
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = j;
            }
        }
        loss = loss + min_dist;
    }

    // loss = (float) loss / (1<<SCALE_FACTOR);
    // loss = (float) loss / (1<<SCALE_FACTOR);
    return loss;

}


double get_sse_double(int numObjs, int numClusters, int numCoords, double * objects, double * clusters_ref)
{
    double loss = 0.0f;
    for (int i=0; i<numObjs; i++) {
        double min_dist = INFINITY;
        int index = 0;
        for (int j=0; j<numClusters; j++) {
            double dist = 0.0f;
            for (int k=0; k<numCoords; k++){
                double coor_1, coor_2;
                coor_1 = objects [1l * i * numCoords + k];
                coor_2 = clusters_ref [1l * j * numCoords + k];
                dist += (coor_1 - coor_2)*(coor_1 - coor_2);
                
            }
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = j;
            }
        }
        loss = loss + min_dist;
    }
    return loss;

}

void print_label_double(int numObjs, int numClusters, int numCoords, double * objects, double * clusters_ref, const char *filename)
{
    ofstream outfile;
    outfile.open(filename, ios::out); 
    if(!outfile.is_open()){
        cout << "Open file failure" << endl;
    }
    else{
        for (int i=0; i<numObjs; i++) {
            double min_dist = INFINITY;
            int index = 0;
            for (int j=0; j<numClusters; j++) {
                double dist = 0;
                for (int k=0; k<numCoords; k++){
                    double coor_1, coor_2;
                    coor_1 = objects [i*numCoords+k];
                    coor_2 = clusters_ref [j*numCoords+k];
                    dist += (coor_1 - coor_2)*(coor_1 - coor_2); 
                }
                if (dist < min_dist) { /* find the min and its array index */
                    min_dist = dist;
                    index = j;
                }
            }
            outfile<<index<<endl;  //在result.txt中写入结果
        }
    }
    outfile.close();
}


void initial_centroids(int numClusters, int numCoords, int numObjs, float* cluster, float* objects) // numCoords��ά������numObjs����������
{
    for (int i=0; i<numClusters; i++) {
        int n = (numObjs/numClusters) * i;

        for (int j=0; j<numCoords; j++)
        {
            cluster[1l * i * numCoords + j] = objects[1l * n * numCoords + j];

            // printf("%f ", cluster[i*numCoords+j]);
        }
        // printf("\n");
    }
}

void initial_centroids_double(int numClusters, int numCoords, int numObjs, double* cluster, double* objects) // numCoords��ά������numObjs����������
{
    for (int i=0; i<numClusters; i++) {
        int n = (numObjs/numClusters) * i;

        for (int j=0; j<numCoords; j++)
        {
            cluster[1l * i * numCoords + j] = objects[1l * n * numCoords + j];

            // printf("%f ", cluster[i*numCoords+j]);
        }
        // printf("\n");
    }
}

void normalization(int nfeatures, int npoints, float* features)
{

    printf("\nStart normalization 0 - 1:\n");

    float* dr_a_min    = (float *)malloc(nfeatures*sizeof(float)); //to store the minimum value of features.....
    float* dr_a_max    = (float *)malloc(nfeatures*sizeof(float)); //to store the miaximum value of features.....

    for (int j = 0; j < nfeatures; ++j)
    {
        float amin = numeric_limits<float>::max();
        float amax = numeric_limits<float>::min();

        for (int i = 0; i < npoints; ++i)
        {
            float a_here = features[1l * i * nfeatures + j];
            if (a_here > amax)
                amax = a_here;
            if (a_here < amin)
                amin = a_here;
        }
        dr_a_min[j]  = amin; //set to the global variable for pm
        dr_a_max[j]  = amax;
        //printf("column: %d, min:%f, max:%f\n", j, amin, amax);
        float arange = amax - amin;
        if (arange > 0)
        {
            for (int i = 0; i < npoints; ++i)
            {
                float tmp = ((features[1l * i * nfeatures + j]-amin)/arange);
                features[1l * i * nfeatures + j] = tmp;
            }
        }
    }

    // for (int i = 0; i < npoints; ++i)
 //    {
 //     for (int j = 0; j < nfeatures; ++j)
 //     {
 //         printf("%f ", normalized_features[i*nfeatures+j]);
 //     }
 //     printf("\n");
 //    }

    printf("normalization finished\n");
}

void normalization_double(int nfeatures, int npoints, double* features)
{

    printf("\nStart normalization 0 - 1:\n");

    double* dr_a_min    = (double *)malloc(nfeatures*sizeof(double)); //to store the minimum value of features.....
    double* dr_a_max    = (double *)malloc(nfeatures*sizeof(double)); //to store the miaximum value of features.....

    for (int j = 0; j < nfeatures; ++j)
    {
        double amin = numeric_limits<double>::max();
        double amax = numeric_limits<double>::min();

        for (int i = 0; i < npoints; ++i)
        {
            double a_here = features[1l * i * nfeatures + j];
            if (a_here > amax)
                amax = a_here;
            if (a_here < amin)
                amin = a_here;
        }
        dr_a_min[j]  = amin; //set to the global variable for pm
        dr_a_max[j]  = amax;
        //printf("column: %d, min:%f, max:%f\n", j, amin, amax);
        double arange = amax - amin;
        if (arange > 0)
        {
            for (int i = 0; i < npoints; ++i)
            {
                double tmp = ((features[1l * i * nfeatures + j]-amin)/arange);
                features[i*nfeatures+j] = tmp;
            }
        }
    }

    // for (int i = 0; i < npoints; ++i)
 //    {
 //     for (int j = 0; j < nfeatures; ++j)
 //     {
 //         printf("%f ", normalized_features[i*nfeatures+j]);
 //     }
 //     printf("\n");
 //    }

    printf("normalization finished\n");
}
