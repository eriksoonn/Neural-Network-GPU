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

/**
 * Initialize NN
*/
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


/**
 * CUDA kernel to forward pass entire batch in parallel (one batch per thread)
*/
__global__ void forward_pass_batch(nn_t *nn, ds_t *ds, double *A, double *Z, int batch_size, int batch_number, int *input_order) {
    unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    unsigned int offset;        // Ofset to first element in tensor
    unsigned int min_batch;     // Index of sample in dataset (across batches)
    unsigned int matrix_size;   // Size of NN (all layers)
    unsigned int input_sample;  // Input sample

    if (thread_id < batch_number) {
        matrix_size = index_counter_1v(nn->layers_size, nn->n_layers);
        
        // For each index in batch, get sample from shuffled dataset and pass through network
        for (int batch_sample_idx = 0; batch_sample_idx < batch_size; batch_sample_idx++) {
            min_batch = thread_id * batch_size + batch_sample_idx;
            input_sample = input_order[min_batch];

            // Offset to store each activation matrix per sample in batch
            offset = thread_id * (batch_size * matrix_size) + batch_sample_idx * matrix_size;
            forward_pass_kernel(nn, &ds->inputs[input_sample * ds->n_inputs], &A[offset], &Z[offset]);
        }      
    }
}

/**
 * GPU function (intended to use in parallel)
 * Pass individual sample through network and compute activations (A and Z)
 * A : activation values
 * B : intermediate values
*/
__device__ void forward_pass_kernel(nn_t *nn, double *input, double *A, double *Z) {
    unsigned int offset_1, offset_2; // Offset value by computing first index to x layer in NN

    // Set input as first activation layer
    for (int i = 0; i < nn->layers_size[0]; i++) {        
        A[i] = input[i];
    }
    
    // Pass previous layer activation values (x - 1) through current layer (x)
    for (int i = 1; i < nn->n_layers; i++) {
        offset_1 = index_counter_1v(nn->layers_size, i);        // Offset to current layer first index
        offset_2 = index_counter_1v(nn->layers_size, i - 1);    // Offset to previous layer first index

        matrix_mul_add(&Z[offset_1], nn->WH[i - 1], &A[offset_2],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        matrix_func(&A[offset_1], &Z[offset_1], nn->layers_size[i], 1, sigmoid);
        matrix_func(&Z[offset_1], &Z[offset_1], nn->layers_size[i], 1, dSigmoid);
    }
}

/**
 * CUDA kernel to nack propagate entire batch in parallel (one batch per thread)
*/
__global__ void back_prop_batch(nn_t *nn, ds_t *ds, double *A, double *Z, double *D, double *d, double *E, double *D_aux, int batch_size, int batch_number, int *input_order, double *loss) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    unsigned int offset_1, offset_2, offset_3, offset_4;
    int min_batch, i;

    if (thread_id < batch_number) {
        for (int sample = 0; sample < batch_size; sample++) {
            min_batch = thread_id * batch_size + sample;
            i = input_order[min_batch];

            offset_1 = thread_id * (batch_size * 101) + sample * 101;
            offset_2 = thread_id * 2410;
            offset_3 = thread_id * 100;
            offset_4 = thread_id * 71;

            back_prop_kernel(nn, &ds->outputs[i * ds->n_outputs], &A[offset_1], &Z[offset_1], &D[offset_2], &d[offset_3], &E[offset_4], &D_aux[offset_2], &loss[thread_id]);
        }    

        loss[thread_id] = loss[thread_id] / batch_size;
    }
}

/**
 * GPU function (intended to use in parallel)
 * Back propagate individual sample through network and compute gradients (D and d)
 * A : activation values
 * B : intermediate values
*/
__device__ void back_prop_kernel(nn_t *nn, double *output, double *A, double *Z, double *D, double *d, double *E, double *D_aux, double *loss) {
    int i, n_l;
    int *l_s;
    double T[600] = {0};
    int offset_1, offset_2, offset_3, offset_4, offset_5, offset_6, offset_7, offset_8, offset_9;

    n_l = nn->n_layers;
    l_s = nn->layers_size;

    offset_1 = index_counter_1v(nn->layers_size, n_l - 1);                              // A Z d --> n_l - 1
    offset_2 = index_counter_1v(nn->layers_size, n_l - 2);                              // A Z d --> n_l - 2
    offset_3 = index_counter_2v(&(nn->layers_size[1]), &(nn->layers_size[0]), n_l - 2); // D D_aux --> n_l - 2
    offset_4 = index_counter_1v(&(nn->layers_size[1]), n_l - 2);                        // E --> n_l - 2

    *loss += mse(&A[offset_1], output, l_s[n_l - 1]);

    matrix_sub(&E[offset_4], &A[offset_1], output, l_s[n_l - 1], 1);
    matrix_mul_dot(&E[offset_4], &E[offset_4], &Z[offset_1], l_s[n_l - 1], 1);  

    matrix_transpose_v2(&A[offset_2], l_s[n_l - 2], 1, T); 
    matrix_mul(&D_aux[offset_3], &E[offset_4], T, l_s[n_l - 1], 1, 1, l_s[n_l - 2]);

    matrix_sum(&D[offset_3], &D[offset_3], &D_aux[offset_3], l_s[n_l - 1], l_s[n_l - 2]);
    matrix_sum(&d[offset_2], &d[offset_2], &E[offset_4], l_s[n_l - 1], 1);

    for (i = n_l - 2; i > 0; i--) {
        offset_5 = index_counter_1v(&(nn->layers_size[1]), i - 1);                          // E --> i - 1
        offset_6 = index_counter_1v(&(nn->layers_size[1]), i);                              // E --> i
        offset_7 = index_counter_1v(nn->layers_size, i);                                    // A Z d --> i - 1
        offset_8 = index_counter_1v(nn->layers_size, i);                                    // A Z d --> i
        offset_9 = index_counter_2v(&(nn->layers_size[1]), &(nn->layers_size[0]), i - 1);   // D D_aux --> i - 1

        matrix_transpose_v2(nn->WH[i], l_s[i + 1], l_s[i], T);
        matrix_mul(&E[offset_5], T, &E[offset_6], l_s[i], l_s[i + 1], l_s[i + 1], 1);

        matrix_mul_dot(&E[offset_5], &E[offset_5], &Z[offset_8], l_s[i], 1);

        matrix_mul(&D_aux[offset_9], &E[offset_5], &A[offset_7], l_s[i], 1, 1, l_s[i - 1]);

        matrix_sum(&D[offset_9], &D[offset_9], &D_aux[offset_9], l_s[i], l_s[i - 1]);
        matrix_sum(&d[offset_7], &d[offset_7], &E[offset_5], l_s[i], 1);
    }
}

/**
 * CUDA kernel to average gradients across batches in parallel (3D Tensor --> 2D Tensor)
*/
__global__ void gradient_average(double *gradient_batch, double *gradient_avg, int size, int batch_number) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int offset;

    if (thread_id < size) {
        for (int j = 0; j < batch_number; j++) {
            offset = (j * size) + thread_id;
            gradient_avg[thread_id] += gradient_batch[offset];
        }
    }
}

/**
 * CUDA kernel to update NN parameters based on gradients (D and d)
*/
__global__ void update(nn_t *nn, double *D, double *d, double lr, int batch_size) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int i, offset;

    if (thread_id == 0) {
        for(i = 0; i < nn->n_layers - 1; i++){
            offset = index_counter_2v(&(nn->layers_size[1]), &(nn->layers_size[0]), i); 

            matrix_mul_cnt(&D[offset], nn->layers_size[i + 1], nn->layers_size[i],  lr * (1.0 / batch_size));
            matrix_mul_cnt(&d[offset], nn->layers_size[i + 1], 1,  lr * (1.0 / batch_size));
            matrix_sub(nn->WH[i], nn->WH[i], &D[offset],  nn->layers_size[i + 1], nn->layers_size[i]);
            matrix_sub(nn->BH[i], nn->BH[i], &d[offset],  nn->layers_size[i + 1], 1);
        }
    } 
}

/**
 * Generate dataset structure in device (GPU) and copy data (from host to device)
*/
void db_to_gpu(int device, ds_t** ds_d, ds_t *ds) {
    cudaSetDevice(device);

    double *inputs_d, *outputs_d, *max_d, *min_d, *mean_d, *std_d;

    cudaMalloc((void**)&(ds_d[device]), sizeof(ds_t)); 
    cudaMemcpy(ds_d[device], ds, sizeof(ds_t), cudaMemcpyHostToDevice); 

    array_to_device(inputs_d, ds->inputs, ds->n_inputs * ds->n_samples);
    array_to_device(outputs_d, ds->outputs, ds->n_outputs * ds->n_samples);
    array_to_device(max_d, ds->max, ds->n_inputs);
    array_to_device(min_d, ds->min, ds->n_inputs);
    array_to_device(mean_d, ds->mean, ds->n_inputs);
    array_to_device(std_d, ds->std, ds->n_inputs);

    cudaMemcpy(&(ds_d[device]->inputs), &inputs_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d[device]->outputs), &outputs_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d[device]->max), &max_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d[device]->min), &min_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d[device]->mean), &mean_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(ds_d[device]->std), &std_d, sizeof(double*), cudaMemcpyHostToDevice);
}

/**
 * Generate NN structure in device (GPU) and copy parameters (from host to device)
*/
void nn_to_gpu(int device, nn_t** nn_d, nn_t *nn) {
    cudaSetDevice(device);

    int *layers_size_d;
    double **WH_d, **BH_d;
        
    cudaMalloc((void**)&(nn_d[device]), sizeof(nn_t)); 
    cudaMemcpy(nn_d[device], nn, sizeof(nn), cudaMemcpyHostToDevice); 

    array_to_device(layers_size_d, nn->layers_size, nn->n_layers);
    matrix_to_device_v1(BH_d, nn->BH, nn->layers_size[1] - 1, &(nn->layers_size[1]), nn->n_layers - 1);
    matrix_to_device_v2(WH_d, nn->WH, nn->layers_size[1] - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), nn->n_layers - 1);

    cudaMemcpy(&(nn_d[device]->layers_size), &layers_size_d, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(nn_d[device]->BH), &BH_d, sizeof(double**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(nn_d[device]->WH), &WH_d, sizeof(double**), cudaMemcpyHostToDevice);
}

void init_weights_gpu(int device, int n_batches, double*& A, double*& Z, double*& D, double*& d, double*& D_aux, double*& E, double*& loss, double*& avg_D, double*& avg_d, double*& loss_h, int size_batch, ds_t *ds, nn_t *nn) {
    cudaSetDevice(device);
    
    int size_1, size_2, size_3, size_4, *order, *order_d;
    
    size_1 = index_counter_1v(nn->layers_size, nn->n_layers);   // Matrix size for A and Z
    size_2 = index_counter_2v(&(nn->layers_size[1]), &(nn->layers_size[0]), nn->n_layers- 1);
    size_3 = index_counter_1v(nn->layers_size, nn->n_layers - 1);
    size_4 = index_counter_1v(&(nn->layers_size[1]), nn->n_layers - 1);

    cudaMalloc(&A, n_batches * size_batch * size_1 * sizeof(double));   // 4D tensor    [num_batches, batch_size, num_layers, layer_size]   (Store activation matrix per sample in batch)
    cudaMalloc(&Z, n_batches * size_batch * size_1 * sizeof(double));   // 4D tensor    [num_batches, batch_size, num_layers, layer_size]   (Store intermediate matrix per sample in batch)
    cudaMalloc(&D, n_batches * size_2 * sizeof(double));                // 3D tensor
    cudaMalloc(&d, n_batches * size_3 * sizeof(double));                // 3D tensor
    cudaMalloc(&D_aux, n_batches * size_2 * sizeof(double));            // 3D tensor
    cudaMalloc(&E, n_batches * size_4 * sizeof(double));                // 3D tensor
    cudaMalloc(&avg_D, size_2 * sizeof(double));                        // 2D tensor
    cudaMalloc(&avg_d, size_3 * sizeof(double));                        // 2D tensor
    cudaMalloc(&loss, n_batches * sizeof(double));                      // 1D tensor

    cudaMemset(A, 0, n_batches * size_batch * size_1 * sizeof(double));
    cudaMemset(Z, 0, n_batches * size_batch * size_1 * sizeof(double));
    cudaMemset(D, 0, n_batches * size_2 * sizeof(double));
    cudaMemset(d, 0, n_batches * size_3 * sizeof(double));
    cudaMemset(D_aux, 0, n_batches * size_2 * sizeof(double));
    cudaMemset(E, 0, n_batches * size_4 * sizeof(double));
    cudaMemset(loss, 0, n_batches * sizeof(double));
    cudaMemset(avg_D, 0, size_2 * sizeof(double));
    cudaMemset(avg_d, 0, size_3 * sizeof(double));

    order = (int*)malloc(ds->n_samples * sizeof(int));
    loss_h = (double*)malloc(n_batches * sizeof(double));

    for (int i = 0; i < ds->n_samples; i++)
        order[i] = i;
}

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr) {
    /*------------------- Multi GPU -------------------*/
    int CUDA_device_count, CPU_thread_count, active_devices;

    cudaGetDeviceCount(&CUDA_device_count);
    CPU_thread_count = omp_get_max_threads();
    active_devices = (CUDA_device_count < CPU_thread_count) ? CUDA_device_count : CPU_thread_count;

    int n_batches = ds->n_samples / size_batch;
    int n_batches_per_device = int(n_batches/CUDA_device_count);

    // Pointers to dataset/NN for each device
    ds_t** ds_d = (ds_t**)malloc(active_devices * sizeof(ds_t*));
    nn_t** nn_d = (nn_t**)malloc(active_devices * sizeof(nn_t*));

    // CUDA kernel launch config
    int thr_col = 32;
    int thr_row = 20;
    int block_col = ceil(double(n_batches) / double(thr_col));
    int block_row = ceil(double(n_batches) / double(thr_row));
    dim3 thr_per_blk(thr_col, thr_row);
    dim3 blk_in_grid(block_col, block_row);

    if (verbose) {
        printf("CUDA devices: %d    CPU threads: %d\n", CUDA_device_count, CPU_thread_count);
        printf("Num of batches: %d  Batches per device: %d\n\n", n_batches, n_batches_per_device);
    }
        
    // Pointers to GPU tensors
    int *order, *order_d;
    double **A, **Z, **D, **d, **D_aux, **E, **loss, **avg_D, **avg_d, **avg_D_h, **avg_d_h, **loss_h, loss_value;

    A = (double**)malloc(CUDA_device_count * sizeof(double*));
    Z = (double**)malloc(CUDA_device_count * sizeof(double*));
    D = (double**)malloc(CUDA_device_count * sizeof(double*));
    d = (double**)malloc(CUDA_device_count * sizeof(double*));
    D_aux = (double**)malloc(CUDA_device_count * sizeof(double*));
    E = (double**)malloc(CUDA_device_count * sizeof(double*));
    loss = (double**)malloc(CUDA_device_count * sizeof(double*));
    avg_D = (double**)malloc(CUDA_device_count * sizeof(double*));
    avg_d = (double**)malloc(CUDA_device_count * sizeof(double*));
    avg_D_h = (double**)malloc(CUDA_device_count * sizeof(double*));
    avg_d_h = (double**)malloc(CUDA_device_count * sizeof(double*));
    loss_h = (double**)malloc(CUDA_device_count * sizeof(double*));

    int size_2 = index_counter_2v(&(nn->layers_size[1]), &(nn->layers_size[0]), nn->n_layers- 1);
    int size_3 = index_counter_1v(nn->layers_size, nn->n_layers - 1);

    /*------------------- Copy dataset/NN to each device and initialize weights  -------------------*/
    #pragma omp parallel for num_threads(active_devices)
    for (int device = 0; device < active_devices; device++) {
        // Copy in parallel (one thread per device)
        db_to_gpu(device, ds_d, ds);
        nn_to_gpu(device, nn_d, nn);

        // Init NN weights and gradients in parallel (one thread per device)
        init_weights_gpu(device, n_batches_per_device, A[device], Z[device], D[device], d[device], D_aux[device], E[device], loss[device], avg_D[device], avg_d[device], loss_h[device], size_batch, ds, nn);
        loss_h[device] = (double*)malloc(n_batches_per_device * sizeof(double));
        avg_D_h[device] = (double*)malloc(size_2 * sizeof(double));
        avg_d_h[device] = (double*)malloc(size_3 * sizeof(double));

        memset(loss_h[device], 0, n_batches_per_device * sizeof(double));
        memset(avg_D_h[device], 0, size_2 * sizeof(double));
        memset(avg_d_h[device], 0, size_3 * sizeof(double));
    }

    // Copy input samples
    order = (int*)malloc(ds->n_samples * sizeof(int));
    for (int i = 0; i < ds->n_samples; i++)
        order[i] = i;

    // Timer (keep track of each epoch)
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*----- Train process -----*/
    for (int n = 0; n < epochs; n++) {

        if(verbose)
            printf("Epoch %d/%d \n", n, epochs);

        // Shuffle input each epoch (inproves generalization of model)
        shuffle(order, ds->n_samples);
        loss_value = 0.0;

        cudaEventRecord(start);

        /*----- Parallel section (thread per device) -----*/
        #pragma omp parallel for num_threads(active_devices)
        for (int device = 0; device < active_devices; device++) {
            cudaSetDevice(device);

            cudaMemset(loss[device], 0, n_batches * sizeof(double));    // Reset loss values
            array_to_device(order_d, order, ds->n_samples);             // Copy shuffled input data to device

            int offset_to_sample = device * (n_batches_per_device * size_batch);

            // Pass batches through NN, back propagate, and compute average gradient across batches
            forward_pass_batch<<<blk_in_grid, thr_per_blk>>>(nn_d[device], ds_d[device], A[device], Z[device], size_batch, n_batches_per_device, &order_d[offset_to_sample]);
            back_prop_batch<<<blk_in_grid, thr_per_blk>>>(nn_d[device], ds_d[device], A[device], Z[device], D[device], d[device], E[device], D_aux[device], size_batch, n_batches_per_device, &order_d[offset_to_sample], loss[device]);
            gradient_average<<<blk_in_grid, thr_per_blk>>>(D[device], avg_D[device], size_2, n_batches_per_device);
            gradient_average<<<blk_in_grid, thr_per_blk>>>(d[device], avg_d[device], size_3, n_batches_per_device);

            // Copy gradients (one per device) to host (to compute average)
            cudaMemcpy(avg_D_h[device], avg_D[device], size_2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(avg_d_h[device], avg_d[device], size_3 * sizeof(double), cudaMemcpyDeviceToHost);
        }

        // Average gradients across devices (supose each device only computes x batches)
        if (active_devices > 1) { 
            array_average_2D(avg_D_h, size_2, active_devices);
            array_average_2D(avg_d_h, size_3, active_devices);
        }

        /*----- Parallel section (thread per device) -----*/
        #pragma omp parallel for num_threads(active_devices)
        for (int device = 0; device < active_devices; device++) {
            cudaSetDevice(device);

            // Copy averaged gradients (across devices) to device
            cudaMemcpy(avg_D[device], avg_D_h[0], size_2 * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(avg_d[device], avg_d_h[0], size_3 * sizeof(double), cudaMemcpyHostToDevice);

            // Update NN based on gradients
            update<<<blk_in_grid, thr_per_blk>>>(nn_d[device], avg_D[device], avg_d[device], lr, n_batches_per_device);

            // Reset gradients
            cudaMemset(D[device], 0, n_batches_per_device * 1 * size_2 * sizeof(double));
            cudaMemset(d[device], 0, n_batches_per_device * 1 * size_3 * sizeof(double));
            memset(avg_D_h[device], 0, size_2 * sizeof(double));
            memset(avg_d_h[device], 0, size_3 * sizeof(double));

            // Copy loss to host
            cudaMemcpy(loss_h[device], loss[device], n_batches_per_device * sizeof(double), cudaMemcpyDeviceToHost);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        for (int device = 0; device < active_devices; device++) {
            loss_value = loss_value + array_sum(loss_h[device], n_batches);
        }
        loss_value = loss_value / n_batches;
        
        if(verbose)
            printf(" time: %f s - loss: %f\n", (milliseconds / 1000), loss_value);
        
        cudaFree(order_d);
    }

    cudaFree(A);
    cudaFree(Z);
    cudaFree(D);
    cudaFree(d);
    cudaFree(D_aux);
    cudaFree(E);
    cudaFree(loss);
    cudaFree(avg_D);
    cudaFree(avg_d);

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

