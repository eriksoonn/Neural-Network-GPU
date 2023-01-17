#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "utils.h"
#include "nn_aux.h"
#include "ds.h"
#include "globals.h"

int seed;
int train_mode;;
int test_mode;
int n_layers;
int verbose;
int epochs;
int batches;
double lr;
int *layers; 
char dataset[100];
char scaling[50];
char model[100];

char *strremove(char *str, const char *sub) {
    char *p, *q, *r;
    if ((q = r = strstr(str, sub)) != NULL) {
        size_t len = strlen(sub);
        while ((r = strstr(p = r + len, sub)) != NULL) {
            memmove(q, p, r - p);
            q += r - p;
        }
        memmove(q, p, strlen(p) + 1);
    }
    return str;
}

int parse_layers(char *optarg){

    int i = 0;
    int n_layers = 0;
    char *end_ptr;
    char *optarg_aux = strdup(optarg);
    char *pt = strtok (optarg_aux,",");
    while (pt != NULL) {
        n_layers++;
        pt = strtok (NULL, ",");
    }

    layers = (int*)malloc(n_layers * sizeof(int));

    pt = strtok (optarg,",");
    while (pt != NULL) {
        layers[i++] = strtol(pt, &end_ptr, 10);
        pt = strtok (NULL, ",");
    }
    return(n_layers);
}

void parse_scaling(ds_t *ds, char *optarg){

    if(strcmp(optarg, "normalization") == 0){
        
        ds->data_scaling = data_normalization;
    }
    else if(strcmp(optarg, "standarization") == 0){
        
        ds->data_scaling = data_standarization;
    }
    else{
        ds->data_scaling = data_zero;
    }

}

void default_options(){

    verbose = 0;
    train_mode = 0;
    test_mode = 0;
    seed = 0;
}

void parse_arguments(int argc, char **argv){

    int c, option_index;
    char *end_ptr;

    default_options();

    while(1){

        static struct option long_options[] =
            {
                {"verbose", no_argument, &verbose, 1},
                {"train", no_argument, &train_mode, 1},
                {"test", no_argument, &test_mode, 1},
                {"seed", required_argument, 0, 's'},
                {"layers", required_argument, 0, 0},
                {"epochs", required_argument, 0, 0},
                {"batch_number", required_argument, 0, 0},
                {"learning_rate", required_argument, 0, 0},
                {"dataset", required_argument, 0, 0},
                {"scaling", required_argument, 0, 0},
                {"model", required_argument, 0, 0},
                //{"weight_init", required_argument, 0, 0},
                //{"loss", required_argument, 0, 0},
                //{"activation", required_argument, 0, 0},
                //{"metric", required_argument, 0, 0},
                {0, 0, 0, 0}
            };

       option_index = 0;

       c = getopt_long (argc, argv, "vs:l:t", long_options, &option_index);

       if (c == -1)
           break;

       switch (c){
           case 0:
               if (long_options[option_index].flag != 0)
                   break;

               if (strcmp(long_options[option_index].name, "learning_rate") == 0){
                    lr = strtod(optarg, &end_ptr);
                    break;
               }
                
               if (strcmp(long_options[option_index].name, "epochs") == 0){
                    epochs = strtol(optarg, &end_ptr, 10);
                    break;
               }
                
               if (strcmp(long_options[option_index].name, "batch_number") == 0){
                    batches = strtol(optarg, &end_ptr, 10);  
                    break;
               }
                
               if (strcmp(long_options[option_index].name, "dataset") == 0){
                    strncpy(dataset, optarg, strlen(optarg));
                    break;
               }
                
               if (strcmp(long_options[option_index].name, "model") == 0){
                    strncpy(model, optarg, strlen(optarg));
                    break;
               }
                
               if (strcmp(long_options[option_index].name, "scaling") == 0){
                    strncpy(scaling, optarg, strlen(optarg));
                    break;
               }
               /*
               if (strcmp(long_options[option_index].name, "weight_init") == 0){
                    strncpy(dataset, optarg, strlen(optarg));
                    break;
               }
               
               if (strcmp(long_options[option_index].name, "loss") == 0){
                    strncpy(dataset, optarg, strlen(optarg));
                    break;
               }

               if (strcmp(long_options[option_index].name, "activation") == 0){
                    strncpy(dataset, optarg, strlen(optarg));
                    break;
               }
            
               if (strcmp(long_options[option_index].name, "metric") == 0){
                    strncpy(dataset, optarg, strlen(optarg));
                    break;
               }
               */
               if (strcmp(long_options[option_index].name, "layers") == 0){
                   n_layers = parse_layers(optarg); 
                   break;
               }

               break;

           case 'v':
               puts ("option -v\n");
               break;

            case 's':
                seed = strtol(optarg, &end_ptr, 10);
                break;

            case '?':
                break;

            default:
                abort ();
        }
    }

  /* Print any remaining command line arguments (not options). */
  if (optind < argc)
    {
      printf ("non-option ARGV-elements: ");
      while (optind < argc)
        printf ("%s ", argv[optind++]);
      putchar ('\n');
    }
}

long diff_time(const struct timespec t2, const struct timespec t1){

    return ((long)t2.tv_sec - (long)t1.tv_sec) * (long)1000000
         + ((long)t2.tv_nsec - (long)t1.tv_nsec) / 1000;
}






