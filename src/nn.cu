#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <math.h>
#include <time.h>
#include "ds.h"
#include "nn.cuh"
#include "nn_aux.cuh"
#include "utils.h"
#include "matrix.cuh"
#include "test.h"
#include "train.h"
#include "globals.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <omp.h>

void init_nn(nn_t *nn, int n_layers, int *layers_size){
    int i;

    nn->n_layers = n_layers;
    nn->layers_size = layers_size;
    nn->init_weight_ptr = init_weight_rnd;
    nn->activation_ptr= (activation_ptr_t*)malloc((nn->n_layers - 1) * sizeof(activation_ptr_t));
    nn->dactivation_ptr= (activation_ptr_t*)malloc((nn->n_layers - 1) * sizeof(activation_ptr_t));
    for(i = 0; i < n_layers - 1; i++){
        nn->activation_ptr[i] = sigmoid;
        nn->dactivation_ptr[i] = dSigmoid;
    }
    nn->loss = mse;
    nn->BH = alloc_matrix_1v(n_layers - 1, &layers_size[1], nn->init_weight_ptr);
    nn->WH = alloc_matrix_2v(n_layers - 1, &layers_size[1], &layers_size[0], nn->init_weight_ptr);
    
}

__global__ void forward_pass_kernel(nn_t *nn, ds_t *ds, double ***A, double ***Z, int batch_num, int *order){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int index = order[threadId];
    //double *inputa = &ds->inputs[index * ds->n_inputs];
    double *inputa = &ds->inputs[1 * ds->n_inputs];

    if (threadId < batch_num) {
        for(int i = 0; i < nn->layers_size[0]; i++){
            A[threadId][0][i] = inputa[i];
        }
    
    
        for(int i = 1; i < nn->n_layers; i++){
            matrix_mul_add(Z[threadId][i], nn->WH[i - 1], A[threadId][i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
            matrix_func(A[threadId][i], Z[threadId][i], nn->layers_size[i], 1, sigmoid);
            matrix_func(Z[threadId][i], Z[threadId][i], nn->layers_size[i], 1, dSigmoid);
        }
    
    }
}

//fix output 0 error!!
__global__ void back_prop_kernel(nn_t *nn, double *output, double ***A, double ***Z, double ***D, double ***d, double ***E, double ***D_aux){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    int i, n_l;
    int *l_s;
    double loss;
    double T[600] = {0};

    n_l = nn->n_layers;
    l_s = nn->layers_size;

    int *size = &(l_s[1]);
    int *size_prev = &(l_s[0]);
    for (int u = 0; u < n_l - 1; u++){
        for (int v = 0; v < size[u] * size_prev[u]; v++){
            D_aux[threadId][u][v] = init_zero();
        }
        for (int v = 0; v < size[u]; v++){
            E[threadId][u][v] = init_zero();
        }
    }

    //loss = nn->loss(A[threadId][n_l - 1], output, l_s[n_l - 1]);
    //matrix_sum(E[threadId][n_l - 2], A[threadId][n_l - 1], A[threadId][n_l - 1], l_s[n_l - 1], 1);
    matrix_sub(E[threadId][n_l - 2], A[threadId][n_l - 1], output, l_s[n_l - 1], 1);
    matrix_mul_dot(E[threadId][n_l - 2], E[threadId][n_l - 2], Z[threadId][n_l - 1], l_s[n_l - 1], 1);  

    if (threadId == 0) {
        printf("OUT:\n");
        for (int i = 0; i < 10; i++) {
            printf(" %f", output[i]);
        }
        printf("\n");

        printf("Z[%d][%d]:\n", threadId, n_l - 1);
        for (int i = 0; i < 10; i++) {
            printf(" %f", Z[threadId][n_l - 1][i]);
        }
        printf("\n");

        printf("A[%d][%d]:\n", threadId, n_l - 1);
        for (int i = 0; i < 10; i++) {
            printf(" %f", A[threadId][n_l - 1][i]);
        }
        printf("\n");

        printf("E[%d][%d]:\n", threadId, n_l - 2);
        for (int i = 0; i < 10; i++) {
            printf(" %f", E[threadId][n_l - 2][i]);
        }
        printf("\n");
    }

    matrix_transpose_v2(A[threadId][n_l - 2], l_s[n_l - 2], 1, T); 
    matrix_mul(D_aux[threadId][n_l - 2], E[threadId][n_l - 2], T, l_s[n_l - 1], 1, 1, l_s[n_l - 2]);
    //matrix_free(T);

    matrix_sum(D[threadId][n_l - 2], D[threadId][n_l - 2], D_aux[threadId][n_l - 2], l_s[n_l - 1], l_s[n_l - 2]);
    matrix_sum(d[threadId][n_l - 2], d[threadId][n_l - 2], E[threadId][n_l - 2], l_s[n_l - 1], 1);

    for (i = n_l - 2; i > 0; i--) {
        matrix_transpose_v2(nn->WH[i], l_s[i + 1], l_s[i], T);
        matrix_mul(E[threadId][i - 1], T, E[threadId][i], l_s[i], l_s[i + 1], l_s[i + 1], 1);
        //matrix_free(T);

        matrix_mul_dot(E[threadId][i - 1], E[threadId][i - 1], Z[threadId][i], l_s[i], 1);

        matrix_mul(D_aux[threadId][i - 1], E[threadId][i - 1], A[threadId][i - 1], l_s[i], 1, 1, l_s[i - 1]);

        matrix_sum(D[threadId][i - 1], D[threadId][i - 1], D_aux[threadId][i - 1], l_s[i], l_s[i - 1]);
        matrix_sum(d[threadId][i - 1], d[threadId][i - 1], E[threadId][i - 1], l_s[i], 1);
    }
}


__global__ void testZ(double ***A, double ***Z) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadId == 1) {
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[1][0][k]);
        }
        printf("\n");
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[7641][0][k]);
        }
        printf("\n");
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[1782][0][k]);
        }
        printf("\n");
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[17082][0][k]);
        }
        printf("\n");
    }
}

__global__ void testZ2(double ***A, double ***Z) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadId == 1) {
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[8402][1][k]);
        }
        printf("\n");
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[7641][1][k]);
        }
        printf("\n");
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[1782][1][k]);
        }
        printf("\n");
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[17082][1][k]);
        }
        printf("\n");
    }
}

template<typename T>
void array_to_device(T *&device, T *host, size_t size) {
    cudaMalloc((void**)&device, size * sizeof(T));
    cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void matrix_to_device(T *&device, T *host, size_t col_size, int *row_sizes, int layers) {
    cudaMalloc((void***)(&device), col_size * sizeof(T));

    for (int i = 0; i < layers; i++) {
        double *array;
        array_to_device(array, host[i], row_sizes[i]);
        cudaMemcpy(device + i, &array, sizeof(T*), cudaMemcpyHostToDevice);
    }
}

template<typename T>
void matrix_to_device2(T *&device, T *host, size_t col_size, int *row_sizes, int *row_sizes_prev, int layers) {
    cudaMalloc((void***)(&device), col_size * sizeof(T));

    for (int i = 0; i < layers; i++) {
        double *array;
        array_to_device(array, host[i], row_sizes[i] * row_sizes_prev[i]);
        cudaMemcpy(device + i, &array, sizeof(T*), cudaMemcpyHostToDevice);
    }
}

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr) {
    /*------------------- Copy data to device -------------------*/
    ds_t *ds_d;
    double *inputs_d, *outputs_d, *max_d, *min_d, *mean_d, *std_d;

    cudaMalloc((void**)&ds_d, sizeof(ds_t)); 
    cudaMemcpy(ds_d, ds, sizeof(ds_t), cudaMemcpyHostToDevice); 
    array_to_device(inputs_d, ds->inputs, ds->n_inputs * ds->n_samples);
    array_to_device(outputs_d, ds->outputs, ds->n_outputs * ds->n_samples);
    array_to_device(max_d, ds->max, ds->n_inputs);
    array_to_device(min_d, ds->min, ds->n_inputs);
    array_to_device(mean_d, ds->mean, ds->n_inputs);
    array_to_device(std_d, ds->std, ds->n_inputs);

    cudaMemcpy(&(ds_d->inputs), &inputs_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->outputs), &outputs_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->max), &max_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->min), &min_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->mean), &mean_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->std), &std_d, sizeof(double*), cudaMemcpyHostToDevice);

    /*------------------- Copy NN to device -------------------*/
    nn_t *nn_d;
    int *layers_size_d;
    double **WH_d, **BH_d;

    cudaMalloc((void**)&nn_d, sizeof(nn_t)); 
    cudaMemcpy(nn_d, nn, sizeof(nn), cudaMemcpyHostToDevice); 
    array_to_device(layers_size_d, nn->layers_size, nn->n_layers);
    matrix_to_device(BH_d, nn->BH, nn->layers_size[1] - 1, &(nn->layers_size[1]), nn->n_layers - 1);
    matrix_to_device2(WH_d, nn->WH, nn->layers_size[1] - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), nn->n_layers - 1);

    cudaMemcpy(&(nn_d->layers_size), &layers_size_d, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(nn_d->BH), &BH_d, sizeof(double**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(nn_d->WH), &WH_d, sizeof(double**), cudaMemcpyHostToDevice);

    /*----- Initialize weights and gradients in devices -----*/
    double ***A, ***Z, ***D, ***d, ***D_aux, ***E, ***A_d, ***Z_d, ***D_d, ***d_d, ***D_aux_d, ***E_d, **temp;
    A = (double ***)malloc(20000 * sizeof(double **));
    Z = (double ***)malloc(20000 * sizeof(double **));
    D = (double ***)malloc(20000 * sizeof(double **));
    d = (double ***)malloc(20000 * sizeof(double **));
    D_aux = (double ***)malloc(20000 * sizeof(double **));
    E = (double ***)malloc(20000 * sizeof(double **));
    for (int i = 0; i < 20000; i++) {
        A[i] = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
        Z[i] = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
        D[i] = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero);
        d[i] = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);
        D_aux[i] = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero);
        E[i] = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);
    }

    cudaMalloc((void****)(&A_d), sizeof(double**) * 20000);
    for (int j = 0; j < 20000; j++) {
        cudaMalloc((void***)(&temp), sizeof(double*) * nn->n_layers);
        for(int i = 0; i < nn->n_layers; i++) {
            double *array;
            cudaMalloc((void**)&(array), sizeof(double) * nn->layers_size[i]);
            cudaMemcpy(array, A[i], sizeof(double) * nn->layers_size[i], cudaMemcpyHostToDevice);
            cudaMemcpy(temp + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(A_d + j, &temp, sizeof(double**), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void****)(&Z_d), sizeof(double**) * 20000);
    for (int j = 0; j < 20000; j++) {
        cudaMalloc((void***)(&temp), sizeof(double*) * nn->n_layers);
        for(int i = 0; i < nn->n_layers; i++) {
            double *array;
            cudaMalloc((void**)&(array), sizeof(double) * nn->layers_size[i]);
            cudaMemcpy(array, Z[i], sizeof(double) * nn->layers_size[i], cudaMemcpyHostToDevice);
            cudaMemcpy(temp + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(Z_d + j, &temp, sizeof(double**), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void****)(&d_d), sizeof(double**) * 20000);
    int *size_l = &(nn->layers_size[1]);
    for (int j = 0; j < 20000; j++) {
        cudaMalloc((void***)(&temp), sizeof(double*) * nn->n_layers - 1);
        for(int i = 0; i < nn->n_layers - 1; i++) {
            double *array;
            cudaMalloc((void**)&(array), sizeof(double) * size_l[i]);
            cudaMemcpy(array, d[i], sizeof(double) * size_l[i], cudaMemcpyHostToDevice);
            cudaMemcpy(temp + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_d + j, &temp, sizeof(double**), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void****)(&D_d), sizeof(double**) * 20000);
    int *size_r = &(nn->layers_size[1]);
    int *size_j = &(nn->layers_size[0]);
    for (int j = 0; j < 20000; j++) {
        cudaMalloc((void***)(&temp), sizeof(double*) * nn->n_layers - 1);
        for(int i = 0; i < nn->n_layers - 1; i++) {
            double *array;
            cudaMalloc((void**)&(array), sizeof(double) * size_r[i] * size_j[i]);
            cudaMemcpy(array, D[i], sizeof(double) * size_r[i] * size_j[i], cudaMemcpyHostToDevice);
            cudaMemcpy(temp + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(D_d + j, &temp, sizeof(double**), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void****)(&E_d), sizeof(double**) * 20000);
    for (int j = 0; j < 20000; j++) {
        cudaMalloc((void***)(&temp), sizeof(double*) * nn->n_layers - 1);
        for(int i = 0; i < nn->n_layers - 1; i++) {
            double *array;
            cudaMalloc((void**)&(array), sizeof(double) * size_l[i]);
            cudaMemcpy(array, E[i], sizeof(double) * size_l[i], cudaMemcpyHostToDevice);
            cudaMemcpy(temp + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(E_d + j, &temp, sizeof(double**), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void****)(&D_aux_d), sizeof(double**) * 20000);
    for (int j = 0; j < 20000; j++) {
        cudaMalloc((void***)(&temp), sizeof(double*) * nn->n_layers - 1);
        for(int i = 0; i < nn->n_layers - 1; i++) {
            double *array;
            cudaMalloc((void**)&(array), sizeof(double) * size_r[i] * size_j[i]);
            cudaMemcpy(array, D[i], sizeof(double) * size_r[i] * size_j[i], cudaMemcpyHostToDevice);
            cudaMemcpy(temp + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(D_aux_d + j, &temp, sizeof(double**), cudaMemcpyHostToDevice);
    }

    printf("ON DEVICE\n");
    int n_batches = ds->n_samples / size_batch;
    printf("%d\n", n_batches);
    int *order;
    order = (int*)malloc(ds->n_samples * sizeof(int));
    for(int i = 0; i < ds->n_samples; i++)
        order[i] = i;
    //shuffle(order, ds->n_samples);
    int *order_d;
    array_to_device(order_d, order, ds->n_samples);


    int *l_s = nn->layers_size;
    int n_l = nn->n_layers;
    printf("ss: %d\n", l_s[n_l - 2] * 1);
    for (int k = n_l - 2; k > 0; k--) {
        printf("ss: %d\n", l_s[k + 1] * l_s[k]);
    }
    printf("out: %d\n", ds->n_outputs * ds->n_samples);

    



    printf("OUT host:\n");
    for (int i = 0; i < 10; i++) {
        printf(" %f", ds->outputs[i]);
    }
    printf("\n");







    int block_col = 1;
    int reg = 625;
    int block_row = ceil(double(n_batches) / double(reg));
    dim3 thr_per_blk(1, reg);
    dim3 blk_in_grid(block_col, block_row);

    printf("Grid : {%d, %d} blocks. Blocks : {%d, %d} threads.\n", blk_in_grid.x, blk_in_grid.y, thr_per_blk.x, thr_per_blk.y);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    printf("START!\n");
    forward_pass_kernel<<<blk_in_grid, thr_per_blk>>>(nn_d, ds_d, A_d, Z_d, n_batches, order_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kelnel run time: %f s\n", (milliseconds / 1000));
    
    cudaDeviceSynchronize();
    printf("FD PASSED!\n");



    //MEMORY LEAK ERROR !!!
/*     double **output_mat;
    cudaMalloc((void***)(&output_mat), 199493 * sizeof(double));
    for (int i = 0; i < 199493; i++) {
        double *array;
        array_to_device(array, outputs_d, ds->n_outputs * ds->n_samples);
        cudaMemcpy(output_mat + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
    } */
    
    //back_prop_kernel<<<blk_in_grid, thr_per_blk>>>(nn_d, &outputs_d[10 * ds->n_outputs], A_d, Z_d, D_d, d_d, E_d, D_aux_d);
    printf("BP PASSED!\n");

    cudaDeviceSynchronize();
    printf("\n");
    testZ2<<<blk_in_grid, thr_per_blk>>>(A_d, Z_d);
    cudaDeviceSynchronize();
    printf("\n");
    testZ<<<blk_in_grid, thr_per_blk>>>(d_d, D_d);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    } else {
        printf("All correct!\n");
    }
    

    sleep(20000);
}














void print_nn(nn_t *nn){

    int i, j, k;
    
    printf("Layers (I/H/O)\n");

    for (i = 0; i < nn->n_layers; i++) {
        printf("%d ", nn->layers_size[i]);
    }
    printf("\n");
    
    printf("Hidden Biases\n ");

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            printf("%lf ", nn->BH[i][j]);
        }
        printf("\n");
    }

    printf("Hidden Weights\n ");
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                printf("%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            printf("\n");
        }
    }

}

void import_nn(nn_t *nn, char *filename){

    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename,"r")) == NULL){
        perror("Error importing the model\n");
        exit(1);
    }
    
    fscanf(fd, "%d ", &n_layers);

    layers = (int*)malloc(n_layers * sizeof(int));

    for (i = 0; i < n_layers; i++) {
        fscanf(fd, "%d ", &(layers[i]));
    }

    init_nn(nn, n_layers, layers);
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fscanf(fd, "%lf ", &(nn->BH[i][j]));
        }
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fscanf(fd, "%lf ", &(nn->WH[i][(j * nn->layers_size[i]) + k]));
            }
        }
    }
    fclose(fd);
}

void export_nn(nn_t *nn, char *filename){

    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename,"w")) == NULL){
        perror("Error exporting the model");
        exit(1);
    }
    
    fprintf(fd, "%d\n", nn->n_layers);

    for (i = 0; i < nn->n_layers; i++) {
        fprintf(fd, "%d ", nn->layers_size[i]);
    }
    fprintf(fd, "\n");
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fprintf(fd, "%lf ", nn->BH[i][j]);
        }
        fprintf(fd, "\n");
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fprintf(fd, "%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            fprintf(fd, "\n");
        }
    }
    fclose(fd);
}

