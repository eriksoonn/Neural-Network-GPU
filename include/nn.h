#ifndef __NN_H
#define __NN_H

#include "ds.h"

typedef double (*activation_ptr_t)(double);

typedef struct nn_t{

    int n_layers;
    int *layers_size;

    double **WH;
    double **BH;

    double (*loss)(double *a, double *output, int length);
    double (*init_weight_ptr)(void);
    activation_ptr_t *activation_ptr;
    activation_ptr_t *dactivation_ptr;

} nn_t;
   
void init_nn(nn_t *nn, int n_layers, int *layers_size); 

void train(nn_t *nn, ds_t *ds, int epochs, int batches, double lr);

void test(nn_t *nn, ds_t *ds);

void import_nn(nn_t *nn, char *filename);

void export_nn(nn_t *nn, char *filename);

void print_nn(nn_t *nn);

void print_deltas(nn_t *nn);

#endif
