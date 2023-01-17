#include "nn.h"
#include "matrix.h"
#include "test.h"
    
void forward_pass_test(nn_t *nn, double *input, double **A){

    int i;

    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
    
    for(i = 1; i < nn->n_layers; i++){

        matrix_mul_add(A[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        matrix_func(A[i], A[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
    }
}

//printf("Expected: %f , Obtained: %f Loss %f\n", output[0], A[nn->n_layers - 1][0], loss);


float precision(int tp, int fp){

    float precision = 0.0;

    return(precision);

}

float recall(int tp, int fn){

    float recall = 0.0;

    return(recall);

}

float f1(float p, float r){

    float f1 = 0.0;

    return(f1);
}


