#include <cuda_runtime_api.h>
#include <cuda.h>

#ifndef __MATRIX_H
#define __MATRIX_H

template<typename T> void array_to_device(T *&device, T *host, size_t size);

double **alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void));

double **alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void));

double *alloc_array(int length);

double *alloc_matrix(int rows, int cols);

void matrix_free_2D(double **m, int n_layers);

void matrix_free(double *m);

__device__ double *m_elem(double *m, int length, int x, int y);

__device__ void matrix_sum(double *c, double *a, double *b, int rows, int cols);

__device__ void matrix_sub(double *c, double *a, double *b, int rows, int cols);

__device__ void matrix_mul_cnt(double *m, int rows, int cols, double cnt);

void matrix_zero(double *m, int rows, int cols);

__device__ void matrix_mul_dot(double *c, double *a, double *b, int rows, int cols);

__device__  double *matrix_transpose_v1(double *m, int rows, int cols);

__device__ void matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

void matrix_mul_trans(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

__device__ void matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d);

__device__ void matrix_func(double *n, double *m, int m_rows, int m_cols, double (*func)(double));

void print_matrix(double *m, int m_rows, int m_cols);

__device__ void *matrix_transpose_v2(double *m, int rows, int cols, double *T);

__host__ __device__ int index_counter_1v(int *sizes, int index);

__host__ __device__ int index_counter_2v(int *sizes, int *sizes_prev, int index);

double array_sum(double *array, int size);

void array_average_2D(double **array_2D, int array_size, int array_number);

template<typename T>
void matrix_to_device_v1(T *&device, T *host, size_t col_size, int *row_sizes, int layers) {
    cudaMalloc((void***)(&device), col_size * sizeof(T));

    for (int i = 0; i < layers; i++) {
        double *array;
        array_to_device(array, host[i], row_sizes[i]);
        cudaMemcpy(device + i, &array, sizeof(T*), cudaMemcpyHostToDevice);
    }
}

template<typename T>
void matrix_to_device_v2(T *&device, T *host, size_t col_size, int *row_sizes, int *row_sizes_prev, int layers) {
    cudaMalloc((void***)(&device), col_size * sizeof(T));

    for (int i = 0; i < layers; i++) {
        double *array;
        array_to_device(array, host[i], row_sizes[i] * row_sizes_prev[i]);
        cudaMemcpy(device + i, &array, sizeof(T*), cudaMemcpyHostToDevice);
    }
}


#endif
