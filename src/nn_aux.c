#include <stdlib.h>
#include <math.h>
#include "ds.h"


/* 
 * Activation functions 
 */

double sigmoid(double x){ 

    return 1 / (1 + exp(-x)); 

}

double dSigmoid(double x){ 
    double sig_z = sigmoid(x);
    return(sig_z * (1 - sig_z));

}

double relu(double x){ 

    return(fmax(0, x)); 

}

double lrelu(double x){ 

    double tmp = x;
    if(x < 0.0)
        tmp = 0.01 * x;
    return(tmp);

}

double dlrelu(double x){ 

    double tmp = 1.0;
    if(x < 0.0)
        tmp = 0.01;
    return(tmp);

}

double drelu(double x){ 

    double tmp = 1.0;
    if(x < 0.0)
        tmp = 0.0;
    return(tmp);

}

double tanh(double x){ 

    return((2/(1 + exp(-2 * x))) - 1);

}

double dtanh(double x){ 

    double sig_z = tanh(x);
    return(1 - (sig_z * sig_z));

}


double soft(double x){ 

    return(log(1 + exp(x)));

}

double dsoft(double x){ 

    return(1 / (1 + exp(-x)));
}

/*
 * Initialization functions 
 */

double init_weight_rnd(){ 

    double w;
    w = ((double)rand())/((double)RAND_MAX);
    return(w); 

}

double init_zero(){ 

    return(0.0); 

}

/* 
 * Loss functions 
 */

double mse(double *a, double *output, int length){

    int i;
    double cost = 0.0;
    
    for(i = 0; i < length; i++){
        cost += ((a[i] - output[i]) * (a[i] - output[i]));
    }
    cost /= length;

    return(cost);

}

double dmse(double *a, double *output, int length){

    int i;
    double cost = 0.0;
    
    for(i = 0; i < length; i++){
        cost += (a[i] - output[i]);
    }
    cost /= length;

    return(cost);

}

/* 
 * Randomize dataset
 */

void shuffle(int *order, int n){
    if (n > 1)
    {
        int i;
        for (i = 0; i < n - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = order[j];
            order[j] = order[i];
            order[i] = t;
        }
    }
}

void data_zero(int n_samples, int n_inputs, double *inputs, double *max, double *min){

}

void data_normalization(int n_samples, int n_inputs, double *inputs, double *max, double *min){

    int i,j;

    for(i = 0; i < n_samples;i++){
        for(j = 0; j < n_inputs;j++){
            inputs[(n_inputs * i) + j] = (inputs[(n_inputs * i) + j] - min[j]) / (max[j] - min[j]);
        }
    }
}

void data_standarization(int n_samples, int n_inputs, double *inputs, double *max, double *min){

    int i,j;

    for(i = 0; i < n_samples;i++){
        for(j = 0; j < n_inputs;j++){
            
            //ds->inputs[i][j] = (ds->inputs[i][j] - ds->min[j]) / (ds->max[j] - ds->min[j]);
        }
    }
}


