#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
// M, N, O represents kernel size, # of channel and output size respectively,
// all represented in terms of multiplications, e.g.: M = 5*5, N = 6, O = 28*28*10

{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias   = NULL;
	weight = NULL;

	// initialize weights and bias
	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N); // biases are identical within the same channel

	cudaMalloc(&weight, sizeof(float) * M * N); // all element position corresponds to a weight

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}


__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *input, float *output, const int N)
// apply step function to input matrices to produce output, N represents number of
// elements of both input and output
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(input[idx]);
	}
}

__global__ void calcLoss(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += dt * grad[idx];
	}
}

// name: concat
// function: concatenate matrices together via channel direction
__global__ void concat(float* output, float* input1, float* input2, float* input3,
						const int size, const int in_channel1, const int in_channel2, const int in_channel3)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int out_channel = in_channel1 + in_channel2 + in_channel3;  // # of channel for output
	const int N = size * size;  // total elements per channel

	if(pos > N){
		for(int n = 0; n < out_channel; n++){
			const int row = pos / size;
			const int col = pos % size;
			if(n < in_channel1){  // first input
				output[(n * size + col) * size + row] = input1[(n * size + col) * size + row];
			}
			else if(n < in_channel2){  // second input
				output[(n * size + col) * size + row] = input2[((n - in_channel1) * size + col) * size + row];
			}
			else{  // last input
				output[(n * size + col) * size + row] = input3[((n - in_channel1 - in_channel2) * size + col) * size + row];
			}
		}
	}
}

// name: fp_conv
// function: convolution layer with padding without stride

__global__ void fp_conv(float* output, float* input, float* weight, float* bias,
						const int kernel_size, const int size, const int in_channel, const int out_channel)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int N = kernel_size * kernel_size * size * size * in_channel * out_channel;  // total number of connections in this convolution
	const int weight_channel = in_channel * out_channel;  // actual number of channels of weight matrix
	const int padding = (kernel_size - 1) / 2;  // number of padding for both ends

	// distribute certain number of connections to each thread regardless of detailed position and shape
	for(int n = N * pos / size; n < N * (pos+1) / size; n++){
		int idx = n;
		const int i_kernel_row = ((idx /= 1	) % kernel_size);  
		const int i_kernel_col = ((idx /= kernel_size	) % kernel_size);
		const int i_channel = ((idx /= kernel_size	) % weight_channel);
		const int i_row = ((idx /= weight_channel	) % size);
		const int i_col = ((idx /= size	) % size);

		// corresponding position of the input matrix
		const int input_row = i_kernel_row + i_row - padding;
		const int input_col = i_kernel_col + i_col - padding;

		if(input_row >= 0 && input_col < size && input_col >=0 && input_col < size){
			atomicAdd(output[((i_channel % out_channel) * size + i_col) * size + i_row], 
						weight[(i_channel * kernel_size + i_kernel_col) * kernel_size + i_kernel_row] 
						* input[((i_channel % in_channel) * size + input_col) * size + input_row]);
		}
	}
}


// __global__ void fp_preact_conv(float input[28][28], float preact[6][24][24], float weight[6][5][5])
// {
// 	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
// 	const int size = blockDim.x * gridDim.x;

// 	const int N = 5*5*6*24*24;

// 	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
// 		int idx = n;
// 		const int i1 = ((idx /= 1	) % 5);
// 		const int i2 = ((idx /= 5	) % 5);
// 		const int i3 = ((idx /= 5	) % 6);
// 		const int i4 = ((idx /= 6	) % 24);
// 		const int i5 = ((idx /= 24	) % 24);

// 		atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
// 	}
// }

// __global__ void fp_bias_conv(float preact[6][24][24], float bias[6])
// {
// 	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
// 	const int size = blockDim.x * gridDim.x;

// 	const int N = 6*24*24;

// 	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
// 		int idx = n;
// 		const int i1 = ((idx /= 1	) % 6);
// 		const int i2 = ((idx /= 6	) % 24);
// 		const int i3 = ((idx /= 24	) % 24);

// 		preact[i1][i2][i3] += bias[i1];
// 	}
// }

__global__ void fp_preact_strideConv(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
// convlolution with strides
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4);
		const int i2 = ((idx /= 4	) % 4);
		const int i3 = ((idx /= 4	) % 6);
		const int i4 = ((idx /= 6	) % 6);
		const int i5 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
	}
}

__global__ void fp_bias_strideConv(float preact[6][6][6], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		preact[i1][i2][i3] += bias[0];
	}
}

__global__ void fp_preact_fc(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
	}
}

__global__ void fp_bias_fc(float preact[10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}

__global__ void bp_weight_fc(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
	}
}

__global__ void bp_bias_fc(float bias[10], float d_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

__global__ void bp_output_strideConv(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
	}
}

__global__ void bp_preact_strideConv(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_strideConv(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

__global__ void bp_bias_strideConv(float bias[1], float d_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
	}
}

__global__ void bp_output_conv(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
	}
}

__global__ void bp_preact_conv(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_conv(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*5*5*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 5);
		const int i3 = ((idx /= 5	) % 5);
		const int i4 = ((idx /= 5	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
	}
}

__global__ void bp_bias_conv(float bias[6], float d_preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
	}
}
