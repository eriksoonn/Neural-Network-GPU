#ifndef __TRAIN_H
#define __TRAIN_H

#include "nn.cuh"
#include "nn_aux.cuh"
#include "ds.h"
#include "matrix.cuh"

double timing_CPU(struct timespec begin, struct timespec end);

void forward_pass(nn_t *nn, double *input, double **A, double **Z);

double back_prop(nn_t *nn, double *output, double **A, double **Z, double **D, double **d);

void update(nn_t *nn, double **D, double **d, double lr, int batch_size);

#endif
