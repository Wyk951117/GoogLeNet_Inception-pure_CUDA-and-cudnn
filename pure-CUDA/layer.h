#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
	public:
	int kernel_size, in_size, out_size, in_channel, out_channel;
	int M, N, O;

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int kernel_size, int in_size, int out_size, int in_channel, int out_channel);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();
};


// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void calcLoss(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void concat(float* output, float* input1, float* input2, float* input3, float* input4, const int size, const int in_channel1, const int in_channel2, const int in_channel3, const int in_channel4);
__global__ void fp_conv(float* output, float* input, float* weight, const int kernel_size, const int size, const int n_size, const int in_channel, const int out_channel, bool SAME);
__global__ void fp_bias_conv(float* preact, float* bias, const int size, const int n_channel);
__global__ void fp_preact_fc(float* input, float* preact, float* weight, const int size, const int in_channel, const int out_channel);
__global__ void fp_bias_fc(float *preact, float *bias, const int n_channel);
__global__ void fp_maxpool(float* output, float* input, const int kernel_size, const int size, const int n_size, const int in_channel, bool SAME);
// Back propagation kernels
__global__ void decat(float* input, float* output1, float* output2, float* output3, float* output4, const int size, const int out_channel1, const int out_channel2, const int out_channel3, const int out_channel4);
__global__ void bp_weight_fc(float *d_weight, float *d_preact, float *p_output, const int size, const int in_channel, const int out_channel);
__global__ void bp_bias_fc(float *bias, float *d_preact, const int n_channel);
__global__ void bp_output_conv(float *d_output, float *weight, float *nd_preact, const int size, const int kernel_size, const int n_size, const int in_channel, const int out_channel, bool CONV, bool SAME);
__global__ void bp_preact_conv(float *d_preact, float *d_output, float *preact, const int size, const int n_channel);
__global__ void bp_weight_conv(float* d_weight, float* d_preact, float* p_output, const int kernel_size, const int size, const int n_size, const int in_channel, const int out_channel, bool SAME);
__global__ void bp_bias_conv(float *bias, float *d_preact, const int size, const int n_channel); 
__global__ void bp_maxpool(float* d_preact, float* preact, float* p_output, const int kernel_size, const int size, const int n_size, const int in_channel, bool SAME);
