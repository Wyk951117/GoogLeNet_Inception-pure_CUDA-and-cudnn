// smaller main for test purpose of single layer or function

#include "layer_demo.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>

// the only five size related parameters needed to test one single layer
int in_size, out_size, in_channel, out_channel, kernel_size;
// input layer doesn't need input related parameters
static Layer input = Layer(0, 0, in_size, 0, in_channel);
static Layer output = Layer(kernel_size, in_size, out_size, in_channel, out_channel);


