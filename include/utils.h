#ifndef _UTILS_H
#define _UTILS_H

#include "ds.h"
#include "globals.h"

char *strremove(char *str, const char *sub);

void parse_arguments(int argc, char **argv);

void parse_scaling(ds_t *ds, char *optarg);

long diff_time(const struct timespec t2, const struct timespec t1);

#endif
