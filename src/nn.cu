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

__global__ void batch_forward_pass(nn_t *nn, ds_t *ds, double *A, double *Z, int batch_size, int batch_number, int *order, int matrix_size) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int min_batch, i;


    if (threadId < batch_number) {
        for (int batch_i = 0; batch_i < batch_size; batch_i++) {
            min_batch = threadId * 10 + batch_i;
            i = order[min_batch];

            unsigned int index = threadId * (batch_size * matrix_size) + batch_i * matrix_size;
            forward_pass_kernel(nn, &ds->inputs[i * ds->n_inputs], &A[index], &Z[index], batch_i);
        }      
    }
}

__device__ void forward_pass_kernel(nn_t *nn, double *input, double *A, double *Z, int sample_in_batch) {
    int j, j2;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    for(int i = 0; i < nn->layers_size[0]; i++){        
        A[i] = input[i];
    }
    
    for(int i = 1; i < nn->n_layers; i++){
        j = index_counter(nn->layers_size, i);
        j2 = index_counter(nn->layers_size, i - 1);
        matrix_mul_add(&Z[j], nn->WH[i - 1], &A[j2],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        matrix_func(&A[j], &Z[j], nn->layers_size[i], 1, sigmoid);
        matrix_func(&Z[j], &Z[j], nn->layers_size[i], 1, dSigmoid);
    }
}

__global__ void check(double *Z, nn_t *nn) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadId == 0) {
        unsigned int index, j;

        printf("batch 1\n");
        for (int i = 0; i < 10; i++) {
            index = 1 * (10 * 101) + i * 101;

            for (int k = 0; k < 10; k++) {
                j = index_counter(nn->layers_size, 3);
                printf("%f ", Z[index + j + k]);
            }
            printf("\n");
        } 

        printf("batch 1278\n");
        for (int i = 0; i < 10; i++) {
            index = 1278 * (10 * 101) + i * 101;

            for (int k = 0; k < 10; k++) {
                j = index_counter(nn->layers_size, 3);
                printf("%f ", Z[index + j + k]);
            }
            printf("\n");
        } 
    }
}

__global__ void batch_back_prop(nn_t *nn, ds_t *ds, double *A, double *Z, double *D, double *d, double *E, double *D_aux, int batch_size, int batch_number, int *order, double *loss) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int min_batch, i;

    if (threadId < batch_number) {
        for (int batch_i = 0; batch_i < batch_size; batch_i++) {
            min_batch = threadId * 10 + batch_i;
            i = order[min_batch];

            unsigned int index = threadId * (batch_size * 101) + batch_i * 101;
            unsigned int index_2 = threadId * 2410;
            unsigned int index_3 = threadId * 100;
            unsigned int index_4 = threadId * 71;

            back_prop_kernel(nn, &ds->outputs[i * ds->n_outputs], &A[index], &Z[index], &D[index_2], &d[index_3], &E[index_4], &D_aux[index_2], &loss[threadId]);
        }      
    }
}

__device__ void back_prop_kernel(nn_t *nn, double *output, double *A, double *Z, double *D, double *d, double *E, double *D_aux, double *loss) {
    int i, n_l;
    int *l_s;
    double T[600] = {0};

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    n_l = nn->n_layers;
    l_s = nn->layers_size;

    //A and Z
    int a_1 = index_counter(nn->layers_size, n_l - 1);  // n_l - 1
    int a_2 = index_counter(nn->layers_size, n_l - 2);  // n_l - 2

    //D and d
    int d_1 = index_counter2(&(nn->layers_size[1]), &(nn->layers_size[0]), n_l - 2);  // n_l - 2

    //E
    int e_1 = index_counter(&(nn->layers_size[1]), n_l - 2);  // n_l - 2

    *loss = mse(&A[a_1], output, l_s[n_l - 1]);

    matrix_sub(&E[e_1], &A[a_1], output, l_s[n_l - 1], 1);
    matrix_mul_dot(&E[e_1], &E[e_1], &Z[a_1], l_s[n_l - 1], 1);  

    matrix_transpose_v2(&A[a_2], l_s[n_l - 2], 1, T); 
    matrix_mul(&D_aux[d_1], &E[e_1], T, l_s[n_l - 1], 1, 1, l_s[n_l - 2]);

    matrix_sum(&D[d_1], &D[d_1], &D_aux[d_1], l_s[n_l - 1], l_s[n_l - 2]);
    matrix_sum(&d[a_2], &d[a_2], &E[e_1], l_s[n_l - 1], 1);

/*     if (threadId == 1) {
        int a_1_1 = index_counter(nn->layers_size, 1);
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[a_1_1 + k]);
        } 
        printf("\n");
    } */

    for (i = n_l - 2; i > 0; i--) {
        int e_3 = index_counter(&(nn->layers_size[1]), i - 1);  // i - 1
        int e_4 = index_counter(&(nn->layers_size[1]), i);      // i
        int e_5 = index_counter(nn->layers_size, i);            // i - 1
        int e_6 = index_counter(nn->layers_size, i);            // i
        int d_2 = index_counter2(&(nn->layers_size[1]), &(nn->layers_size[0]), i - 1);    // i - 1

        matrix_transpose_v2(nn->WH[i], l_s[i + 1], l_s[i], T);
        matrix_mul(&E[e_3], T, &E[e_4], l_s[i], l_s[i + 1], l_s[i + 1], 1);
        //matrix_free(T);

        matrix_mul_dot(&E[e_3], &E[e_3], &Z[e_6], l_s[i], 1);

        matrix_mul(&D_aux[d_2], &E[e_3], &A[e_5], l_s[i], 1, 1, l_s[i - 1]);

        matrix_sum(&D[d_2], &D[d_2], &D_aux[d_2], l_s[i], l_s[i - 1]);
        matrix_sum(&d[e_5], &d[e_5], &E[e_3], l_s[i], 1);
    }
}

__global__ void gradient_average(double *batch_gradient, double *gradient, int size, int batch_num) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int f_index;

    if (threadId < size) {
        for (int j = 0; j < batch_num; j++) {
            f_index = j * size;
            gradient[threadId] += batch_gradient[f_index + threadId];
        }
    }
}

__global__ void update(nn_t *nn, double *D, double *d, double lr, int batch_size){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int i;
    struct timespec begin, end; 
    int d_1;

    if (threadId == 0) {
        for(i = 0; i < nn->n_layers - 1; i++){
            d_1 = index_counter2(&(nn->layers_size[1]), &(nn->layers_size[0]), i); 
            matrix_mul_cnt(&D[d_1], nn->layers_size[i + 1], nn->layers_size[i],  lr * (1.0 / batch_size));
            matrix_mul_cnt(&d[d_1], nn->layers_size[i + 1], 1,  lr * (1.0 / batch_size));
            matrix_sub(nn->WH[i], nn->WH[i], &D[d_1],  nn->layers_size[i + 1], nn->layers_size[i]);
            matrix_sub(nn->BH[i], nn->BH[i], &d[d_1],  nn->layers_size[i + 1], 1);
            //matrix_zero(&D[d_1], nn->layers_size[i + 1], nn->layers_size[i]);
            //matrix_zero(&d[d_1], nn->layers_size[i + 1], 1);
        }
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

    printf("Inputs: %d  Outputs: %d", ds->n_inputs, ds->n_outputs);

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
    double *A, *Z, *D, *d, *D_aux, *E, *loss;
    int size_1 = 0;
    int size_2 = 0;
    int size_3 = 0;
    int size_4 = 0;
    int n_batches = ds->n_samples / size_batch;

    for (int i = 0; i < nn->n_layers; i++) {
        size_1 += nn->layers_size[i];
    }

    for (int i = 0; i < nn->n_layers - 1; i++) {
        size_3 += nn->layers_size[i];
    }

    int *aux_1 = &(nn->layers_size[1]);
    int *aux_0 = &(nn->layers_size[0]);
    for (int i = 0; i < nn->n_layers - 1; i++) {
        size_2 += aux_1[i] * aux_0[i];
    }

    for (int i = 0; i < nn->n_layers - 1; i++) {
        size_4 += aux_1[i];
    }

    printf("Size A Z --> %d\n", size_1);
    printf("Size D D_aux --> %d\n", size_2);
    printf("Size d --> %d\n", size_3);
    printf("Size E --> %d\n", size_4);

    // num batches * samples * layers * layer size
    cudaMalloc(&A, n_batches * 10 * size_1 * sizeof(double));
    cudaMalloc(&Z, n_batches * 10 * size_1 * sizeof(double));

    cudaMalloc(&D, n_batches * 1 * size_2 * sizeof(double));
    cudaMalloc(&d, n_batches * 1 * size_3 * sizeof(double));

    cudaMalloc(&D_aux, n_batches * 1 * size_2 * sizeof(double));
    cudaMalloc(&E, n_batches * 1 * size_4 * sizeof(double));

    cudaMalloc(&loss, n_batches * sizeof(double));

    cudaMemset(A, 0, n_batches * 10 * size_1 * sizeof(double));
    cudaMemset(Z, 0, n_batches * 10 * size_1 * sizeof(double));
    cudaMemset(D, 0, n_batches * 1 * size_2 * sizeof(double));
    cudaMemset(d, 0, n_batches * 1 * size_3 * sizeof(double));
    cudaMemset(D_aux, 0, n_batches * 1 * size_2 * sizeof(double));
    cudaMemset(E, 0, n_batches * 1 * size_4 * sizeof(double));
    cudaMemset(loss, 0, n_batches * sizeof(double));

    /*----- TRAIN -----*/
    int *order, *order_d;
    int block_col = 1;
    int reg = 640;
    int block_row = ceil(double(n_batches) / double(reg));
    dim3 thr_per_blk(1, reg);
    dim3 blk_in_grid(block_col, block_row);
    order = (int*)malloc(ds->n_samples * sizeof(int));
    for(int i = 0; i < ds->n_samples; i++)
        order[i] = i;


    double *avg_D, *avg_d;
    cudaMalloc(&avg_D, size_2 * sizeof(double));
    cudaMalloc(&avg_d, size_3 * sizeof(double));
    cudaMemset(avg_D, 0, size_2 * sizeof(double));
    cudaMemset(avg_d, 0, size_3 * sizeof(double));

    double *loss_h;
    double loss_f;
    loss_h = (double*)malloc(n_batches * sizeof(double));

    for (int n = 0; n < epochs; n++) {
        printf("Epoch %d/%d \n", n, epochs);
        cudaMemset(loss, 0, n_batches * sizeof(double));
        //shuffle(order, ds->n_samples);
        array_to_device(order_d, order, ds->n_samples);

        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        batch_forward_pass<<<blk_in_grid, thr_per_blk>>>(nn_d, ds_d, A, Z, 10, n_batches, order_d, size_1);
        //check<<<blk_in_grid, thr_per_blk>>>(Z, nn_d);

        batch_back_prop<<<blk_in_grid, thr_per_blk>>>(nn_d, ds_d, A, Z, D, d, E, D_aux, 10, n_batches, order_d, loss);
        //CHANGE LOSS TO PER SAMPLE
        cudaMemcpy(loss_h, loss, n_batches * sizeof(double), cudaMemcpyDeviceToHost);
        loss_f = 0;
        for (int i = 0; i < n_batches; i++) {
            loss_f += loss_h[i];
        }
        loss_f = loss_f / n_batches;


        gradient_average<<<blk_in_grid, thr_per_blk>>>(D, avg_D, size_2, n_batches);
        gradient_average<<<blk_in_grid, thr_per_blk>>>(d, avg_d, size_3, n_batches);

        update<<<blk_in_grid, thr_per_blk>>>(nn_d, avg_D, avg_d, 0.01, n_batches);
        cudaMemset(D, 0, n_batches * 1 * size_2 * sizeof(double));
        cudaMemset(d, 0, n_batches * 1 * size_3 * sizeof(double));

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf(" time: %f s - loss: %f\n", (milliseconds / 1000), loss_f);
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

