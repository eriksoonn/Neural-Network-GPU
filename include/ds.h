#ifndef __DS_H
#define __DS_H

#include <stdio.h>

typedef struct ds_t{

    int n_samples;
    int n_inputs;
    int n_outputs;
    double *inputs;
    double *outputs;
    double *max;
    double *min;
    double *mean;
    double *std;
    
    void (*data_scaling)(int, int, double *, double *, double *);

} ds_t;

int check_n_lines(FILE *fd);

int check_n_columns(FILE *fd, const char *sep);

void read_csv(char *filename, ds_t *ds, int inputs, int ouputs);

void print_ds(ds_t *ds);

#endif
