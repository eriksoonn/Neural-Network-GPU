#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "ds.h"
#include "utils.h"

#define LENGTH 4096

int check_n_lines(FILE *fd){
    
    char line[LENGTH];
    int n_lines = 1;

    fgets(line, LENGTH, fd);
    if(line[0] == '"')
        n_lines--;
    while( fgets(line, sizeof(line), fd) != NULL)
        n_lines++;

    fseek(fd, 0, SEEK_SET);

    return(n_lines);
}

int check_n_columns(FILE *fd, const char *sep){
	    
    int n_inputs = 0;
    char line[LENGTH];
    char* tok;

    fgets(line, LENGTH, fd);
    if(line[0] == '"')
        fgets(line, LENGTH, fd);

    tok = strtok(line, sep);
    while (tok != NULL){
        tok = strtok(NULL, sep);
        n_inputs++;
    }

    fseek(fd, -1, SEEK_CUR);

    return(n_inputs);
}

void read_csv(char *filename, ds_t *ds, int inputs, int outputs){
	
	int i, j, n_i_o;
    char line[LENGTH];
    char *tok, *endptr;
    double tok_aux;
    FILE *fd;

    const char sep[2] = ",";

	if ((fd = fopen(filename, "r")) == NULL) {
        perror("Error: ");
        exit(1);
    }

    ds->n_samples = check_n_lines(fd);
    n_i_o = check_n_columns(fd, sep);
    
    if(n_i_o != (inputs + outputs)){
        fprintf(stderr, "Error: Number of inputs (%d) and outputs (%d) of the model differ from those of the dataset (%d).\n", inputs, outputs, n_i_o);
        exit(1);
    }

    ds->n_inputs = inputs;
    ds->n_outputs = outputs;
    
    ds->inputs = (double*)malloc(inputs * ds->n_samples * sizeof(double));
    ds->outputs = (double*)malloc(outputs * ds->n_samples * sizeof(double));
    ds->max = (double*)malloc(inputs * sizeof(double));
    ds->min = (double*)malloc(inputs * sizeof(double));
    ds->std = (double*)malloc(inputs * sizeof(double));
    ds->mean = (double*)malloc(inputs * sizeof(double));
    
    for(i = 0;i < inputs;i++){
        ds->max[i] = -DBL_MAX;   
        ds->min[i] = DBL_MAX;   
    }

    i = 0;
    
    while (fgets(line, LENGTH, fd) != NULL) {
        j = 0;
        //ds->inputs[i] = (double*)malloc(inputs * sizeof(double));
        //ds->outputs[i] = (double*)malloc(outputs * sizeof(double));
        //tmp = strdup(line);

	    tok = strtok(line, sep);
        
        while (tok != NULL){

            tok_aux = strtod(strremove(tok, "\""),  &endptr);
        
            if(j < inputs){
                ds->inputs[(inputs * i) + j] = tok_aux;
                if(tok_aux > ds->max[j])
                    ds->max[j] = tok_aux;
                if(tok_aux < ds->min[j])
                    ds->min[j] = tok_aux;
            }
            else 
                ds->outputs[(inputs * i) + (j - inputs)] = tok_aux;

            tok = strtok(NULL, sep);
            j++;
        }
        i++;
        //free(tmp);
    }

    parse_scaling(ds, scaling);
    ds->data_scaling(ds->n_samples, ds->n_inputs, ds->inputs, ds->max, ds->min);
}

void print_ds(ds_t *ds){

    int i, j;

    for(i = 0; i < ds->n_samples; i++){
        for(j = 0; j < ds->n_inputs; j++){
            printf("%lf ", ds->inputs[(ds->n_inputs * i) + j]);
        }
        printf(" | ");
        for(j = 0; j < ds->n_outputs; j++){
            printf("%lf ", ds->outputs[(ds->n_inputs * i) + j]);
        }
        printf("\n");
    }
}




