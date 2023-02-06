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



/* void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr){

    int i, n, x, n_batches, min_batch;
    double **A, **Z, **D, **d;;
    int *order;
    double loss;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
  
    order = (int*)malloc(ds->n_samples * sizeof(int));

    int tid = omp_get_max_threads();
    printf("Hello World from thread = %d\n", tid);
    
    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); // activations of each layer 
    Z = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); // weighted input of each layer
    D = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero); //gradients of the weights
    d = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero); //gradients of the biases
    
    n_batches = ds->n_samples / size_batch;

    for(i = 0; i < ds->n_samples; i++)
        order[i] = i;
    
    for (n=0; n < epochs;n++) {
            
        if(verbose)
            printf("Epoch %d/%d \n", n, epochs);
        
        loss = 0.0;
        shuffle(order, ds->n_samples);
        
        //paralelize this
        for (x = 0; x < n_batches; x++) {
            //clock_gettime(clk_id, &t1);
            for(min_batch = (x * size_batch); min_batch < ((x + 1) * size_batch); min_batch++){
                if (min_batch < -1)
                    continue;

                i = order[min_batch];
                //printf("A -> %d: %f %f %f %f %f %f\n", min_batch, A[1][1], A[2][1], A[1][5], A[2][5], A[1][7], A[2][7]);
                clock_gettime(clk_id, &t1);
                forward_pass(nn, &ds->inputs[i * ds->n_inputs], A, Z); //A and Z output 
                clock_gettime(clk_id, &t2);
                printf(" time: %f s\n", diff_time(t2, t1)/ 1000000.0);
                loss += back_prop(nn, &ds->outputs[i * ds->n_outputs], A, Z, D, d); //A and Z input D and d output
                
                printf("i -> %d: %f %f %f %f %f %f\n", min_batch, D[1][1], D[2][1], D[1][5], D[2][5], D[1][7], D[2][7]);
            }

            //printf("end: %f %f %f %f %f %f\n", D[1][1], D[2][1], D[1][5], D[2][5], D[1][7], D[2][7]);
            update(nn, D, d, lr, size_batch);
            break;
            //clock_gettime(clk_id, &t2);
            //printf(" time: %f s\n", diff_time(t2, t1)/ 1000000.0);
        }
        break;

        if(verbose)
            printf(" time: %f s - loss: %.*f\n", diff_time(t2, t1)/ 1000000.0, 12, loss / ds->n_samples);

    }

} */

__global__ void forward_pass_kernel(nn_t *nn, double *input, double ***A, double ***Z){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadId < 1000) {
        for(int i = 0; i < nn->layers_size[0]; i++){
            A[threadId][0][i] = input[i];
        }
    
        for(int i = 1; i < nn->n_layers; i++){
            matrix_mul_add(Z[threadId][i], nn->WH[i - 1], A[threadId][i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
            matrix_func(A[threadId][i], Z[threadId][i], nn->layers_size[i], 1, sigmoid);
            matrix_func(Z[threadId][i], Z[threadId][i], nn->layers_size[i], 1, dSigmoid);
        }
    
    }
}

__global__ void testZ(double ***A, double ***Z) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadId == 1) {
        for (int k = 0; k < 10; k++) {
            printf("%f ", A[threadId][0][k]);
        }
        printf("\n");
        for (int k = 0; k < 10; k++) {
            printf("%f ", Z[threadId][1][k]);
        }
        printf("\n");
    }
}

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr) {
    //Copy data to GPU
    ds_t *ds_d;
    double *inputs_d, *outputs_d, *max_d, *min_d, *mean_d, *std_d;

    cudaMalloc((void**)&ds_d, sizeof(ds_t)); 
    cudaMalloc(&inputs_d, ds->n_inputs * ds->n_samples * sizeof(double));
    cudaMalloc(&outputs_d, ds->n_outputs * ds->n_samples * sizeof(double));
    cudaMalloc(&max_d, ds->n_inputs * sizeof(double));
    cudaMalloc(&min_d, ds->n_inputs * sizeof(double));
    cudaMalloc(&mean_d, ds->n_inputs * sizeof(double));
    cudaMalloc(&std_d, ds->n_inputs * sizeof(double));

    cudaMemcpy(ds_d, ds, sizeof(ds_t), cudaMemcpyHostToDevice); 
    cudaMemcpy(inputs_d, ds->inputs, ds->n_inputs * ds->n_samples * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(outputs_d, ds->outputs, ds->n_outputs * ds->n_samples * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(max_d, ds->max, ds->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(min_d, ds->min, ds->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mean_d, ds->mean, ds->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(std_d, ds->max, ds->n_inputs * sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(&(ds_d->inputs), &inputs_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->outputs), &outputs_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->max), &max_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->min), &min_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->mean), &mean_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d->std), &std_d, sizeof(double*), cudaMemcpyHostToDevice);


    nn_t *nn_d;
    int *layers_size_d;
    double **WH_d, **BH_d;

    cudaMalloc((void**)&nn_d, sizeof(nn_t)); 
    cudaMalloc(&layers_size_d, sizeof(int) * nn->n_layers);

    cudaMemcpy(nn_d, nn, sizeof(nn), cudaMemcpyHostToDevice); 
    cudaMemcpy(layers_size_d, nn->layers_size, sizeof(int) * nn->n_layers, cudaMemcpyHostToDevice);

    cudaMemcpy(&(nn_d->layers_size), &layers_size_d, sizeof(int*), cudaMemcpyHostToDevice);






    cudaMalloc((void***)(&BH_d), sizeof(double*) * (nn->layers_size[1] - 1));
    int *sizes = &(nn->layers_size[1]);
    for(int i = 0; i < nn->n_layers - 1; i++) {
        double *array;
        cudaMalloc((void**)&(array), sizeof(double) * sizes[i]);
        cudaMemcpy(array, nn->BH[i], sizeof(double) * sizes[i], cudaMemcpyHostToDevice);
        cudaMemcpy(BH_d + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void***)(&WH_d), sizeof(double*) * (nn->layers_size[1] - 1));
    sizes = &(nn->layers_size[1]);
    int *sizes_prev = &(nn->layers_size[0]);
    for(int i = 0; i < nn->n_layers - 1; i++) {
        double *array;
        cudaMalloc((void**)&(array), sizeof(double) * sizes[i] * sizes_prev[i]);
        cudaMemcpy(array, nn->WH[i], sizeof(double) * sizes[i] * sizes_prev[i], cudaMemcpyHostToDevice);
        cudaMemcpy(WH_d + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(&(nn_d->BH), &BH_d, sizeof(double**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(nn_d->WH), &WH_d, sizeof(double**), cudaMemcpyHostToDevice);

    double ***A, ***Z, ***A_d, ***Z_d, **temp;
    A = (double ***)malloc(1000 * sizeof(double **));
    Z = (double ***)malloc(1000 * sizeof(double **));
    for (int i = 0; i < 1000; i++) {
        A[i] = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
        Z[i] = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
    }

    cudaMalloc((void****)(&A_d), sizeof(double**) * 1000);
    for (int j = 0; j < 1000; j++) {
        cudaMalloc((void***)(&temp), sizeof(double*) * nn->n_layers);
        for(int i = 0; i < nn->n_layers; i++) {
            double *array;
            cudaMalloc((void**)&(array), sizeof(double) * nn->layers_size[i]);
            cudaMemcpy(array, A[i], sizeof(double) * nn->layers_size[i], cudaMemcpyHostToDevice);
            cudaMemcpy(temp + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(A_d + j, &temp, sizeof(double**), cudaMemcpyHostToDevice);
    }

    cudaMalloc((void****)(&Z_d), sizeof(double**) * 1000);
    for (int j = 0; j < 1000; j++) {
        cudaMalloc((void***)(&temp), sizeof(double*) * nn->n_layers);
        for(int i = 0; i < nn->n_layers; i++) {
            double *array;
            cudaMalloc((void**)&(array), sizeof(double) * nn->layers_size[i]);
            cudaMemcpy(array, A[i], sizeof(double) * nn->layers_size[i], cudaMemcpyHostToDevice);
            cudaMemcpy(temp + i, &array, sizeof(double*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(Z_d + j, &temp, sizeof(double**), cudaMemcpyHostToDevice);
    }

    printf("ON DEVICE\n");

    int block_col = ceil(double(1) / double(25));
    int block_row = ceil(double(10000) / double(25));
    dim3 thr_per_blk(25, 25);
    dim3 blk_in_grid(block_col, block_row);
    printf("starting kernel\n");

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    forward_pass_kernel<<<blk_in_grid, thr_per_blk>>>(nn_d, &inputs_d[1 * ds->n_inputs], A_d, Z_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kelnel run time: %f s\n", (milliseconds / 1000));
    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
    } else {
        printf("All correct!\n");
    }
    testZ<<<blk_in_grid, thr_per_blk>>>(A_d, Z_d);

    sleep(1000);
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

