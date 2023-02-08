#include <cuda_runtime_api.h>
#include <cuda.h>

#ifndef __MATRIX_H
#define __MATRIX_H

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

__device__  double *matrix_transpose(double *m, int rows, int cols);

__device__ void matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

void matrix_mul_trans(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

__device__ void matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d);

__device__ void matrix_func(double *n, double *m, int m_rows, int m_cols, double (*func)(double));

void print_matrix(double *m, int m_rows, int m_cols);

__device__ void *matrix_transpose_v2(double *m, int rows, int cols, double *T);

#endif
