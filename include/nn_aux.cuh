#include <cuda_runtime_api.h>
#include <cuda.h>

#ifndef __NN_AUX_H
#define __NN_AUX_H

#include "ds.h"

__host__ __device__ double sigmoid(double x); 

__host__ __device__ double dSigmoid(double x); 

double relu(double x);

double lrelu(double x);

double drelu(double x);

double dlrelu(double x);

double tanh(double x);

double dtanh(double x);

double soft(double x);

double dsoft(double x);

double init_weight_rnd();

__device__ __host__ double init_zero();

void shuffle(int *order, int n);

__device__ double mse(double *a, double *output, int length);

void data_zero(int n_samples, int n_inputs, double *inputs, double *max, double *min);

void data_normalization(int n_samples, int n_inputs, double *inputs, double *max, double *min);

void data_standarization(int n_samples, int n_inputs, double *inputs, double *max, double *min);

#endif
