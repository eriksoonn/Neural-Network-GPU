#ifndef __GLOBALS_H
#define __GLOBALS_H

extern int verbose;
extern int seed;
extern int train_mode;
extern int test_mode;
extern int n_layers;
extern int epochs;
extern int batches;

extern double lr;

extern int *layers;

extern char dataset[100];
extern char scaling[50];
extern char model[100];

#endif
