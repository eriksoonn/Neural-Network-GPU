#ifndef __NN_AUX_H
#define __NN_AUX_H

#include "ds.h"

double sigmoid(double x); 

double dSigmoid(double x); 

double relu(double x);

double lrelu(double x);

double drelu(double x);

double dlrelu(double x);

double tanh(double x);

double dtanh(double x);

double soft(double x);

double dsoft(double x);

double init_weight_rnd();

double init_zero();

void shuffle(int *order, int n);

double mse(double *a, double *output, int length);

void data_zero(int n_samples, int n_inputs, double *inputs, double *max, double *min);

void data_normalization(int n_samples, int n_inputs, double *inputs, double *max, double *min);

void data_standarization(int n_samples, int n_inputs, double *inputs, double *max, double *min);

#endif
