#include "layer.h"
#include <cstdio>

// Constructor
Layer::Layer(int kernel_size, int in_size, int out_size, int in_channel, int out_channel)
// M, N, O represents kernel size, # of channel and output size respectively,
// all represented in terms of multiplications, e.g.: M = 5*5, N = 6, O = 28*28*10

{
	this->kernel_size = kernel_size;
	this->in_size = in_size;
	this->out_size = out_size;
	this->in_channel = in_channel;
	this->out_channel = out_channel;
  
	int N = in_channel * out_channel;
	int M = kernel_size * kernel_size * out_size * out_size;
	int O = out_channel * out_size * out_size;
  
	float *h_bias, *h_weight;
	// host memory allocation
	//h_bias = (float *)malloc(sizeof(float) * N);
	//h_weight = (float *)malloc(sizeof(float) * N * M);
	cudaMallocHost(&h_bias, sizeof(float) * N);
	cudaMallocHost(&h_weight, sizeof(float) * N * M);

	float *output, *preact, *bias, *weight;

	// initialize weights and bias
	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < M; ++j) {
			h_weight[i * N + j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}
  fprintf(stdout, "%d, %d, %d, %d, %d, %d, %d, %d\n", kernel_size, in_size, out_size, in_channel, out_channel, N, M, O);
	// device memory allocation
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

/**name: step_function
 * function: implement sigmoid step function as activation function
 */
__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

/**name: apply_step_function
 * function: apply step function to input matrices to produce output, N represents number of elements of both input and output
 * @param N     total number of elements in both input and output
 */
__global__ void apply_step_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) {
		output[idx] = step_function(input[idx]);
	}
}

__global__ void calcLoss(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) { 
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) {
		output[idx] += dt * grad[idx];
	}
}

/**name: concat
 * function: concatenate matrices together via the direction of channels
 * @param output       output of concat operation
 * @param input1       first input of concat operation, the same for 2,3,4
 * @param in_channel1  the number of channels of the first input
 * @param size         the height and width of each channel (each feature map)
 */

__global__ void concat(float* output, float* input1, float* input2, float* input3, float* input4,
						const int size, const int in_channel1, const int in_channel2, const int in_channel3, const int in_channel4)
{
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int out_channel = in_channel1 + in_channel2 + in_channel3 + in_channel4;  // # of channel for output
	const int N = size * size;  // total elements per channel

	if(pos < N){
		for(int n = 0; n < out_channel; n++){
			const int row = pos / size;
			const int col = pos % size;
			if(n < in_channel1){  // first input
				output[(n * size + col) * size + row] = input1[(n * size + col) * size + row];
			}
			else if(n < in_channel1 + in_channel2){  // second input
				output[(n * size + col) * size + row] = input2[((n - in_channel1) * size + col) * size + row];
			}
			else if(n < in_channel1 + in_channel2 + in_channel3){  // third input
				output[(n * size + col) * size + row] = input3[((n - in_channel1 - in_channel2) * size + col) * size + row];
			}
			else{  // last input
				output[(n * size + col) * size + row] = input4[((n - in_channel1 - in_channel2 - in_channel3) * size + col) * size + row];
			}
		}
	}
}

/**name: decat
 * function: inverse process of concat operation for backpropagation
 * @param input        input of concat operation
 * @param output1      first output of concat operation, the same for 2,3,4
 * @param out_channel1 the number of channels of the first output
 * @param size         the height and width of each channel (each feature map)
 */

 __global__ void decat(float* input, float* output1, float* output2, float* output3, float* output4,
	const int size, const int out_channel1, const int out_channel2, const int out_channel3, const int out_channel4)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int in_channel = out_channel1 + out_channel2 + out_channel3 + out_channel4;  // # of channel of input
	const int N = size * size;  // total elements per channel

	if(pos < N){
		for(int n = 0; n < in_channel; n++){
			const int row = pos / size;
			const int col = pos % size;
			if(n < out_channel1){  // first output
				output1[(n * size + col) * size + row] = input[(n * size + col) * size + row];
			}
			else if(n < out_channel1 + out_channel2){  // second output
				output2[((n - out_channel1) * size + col) * size + row] = input[(n * size + col) * size + row];
			}
			else if(n < out_channel1 + out_channel2 + out_channel3){  // third output
				output3[((n - out_channel1 - out_channel2) * size + col) * size + row] = input[(n * size + col) * size + row];
			}
			else{  // last output
				output4[((n - out_channel1 - out_channel2 - out_channel3) * size + col) * size + row] = input[(n * size + col) * size + row];
			}
		}
	}
}

/**name: fp_conv
 * function: convolution layer with padding without stride
 * @param output           output data matrix of convolution operation
 * @param input            input data matrix of convolution operation
 * @param weight           weight matrix of operation convolution
 * @param kernel_size      the size of weight matrix
 * @param size             the size of data matrix
 * @param n_size           the size of output feature matrix
 * @param in_channel       the number of channels for input data matrix
 * @param out_channel      the number of channels for output data matrix
 * @param SAME          boolean decide whether use "SAME" padding for this convolution operation
 */

__global__ void fp_conv(float* output, float* input, float* weight, const int kernel_size,  
						const int size, const int n_size, const int in_channel, const int out_channel, bool SAME)
{
  printf("entering kernel\n"); 
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;
	const int N = kernel_size * kernel_size * n_size * n_size * in_channel * out_channel;  // total number of connections in this convolution
	const int weight_channel = in_channel * out_channel;  // actual number of channels of weight matrix
	const int padding = (kernel_size - 1) / 2;  // number of padding for both ends
 
	// distribute certain number of connections to each thread regardless of detailed position and shape
	for(int n = N * pos / totalPos; n < N * (pos+1) / totalPos; n++){
		int idx = n;
		const int i_kernel_row = ((idx /= 1	) % kernel_size);  
		const int i_kernel_col = ((idx /= kernel_size	) % kernel_size);
		const int i_channel = ((idx /= kernel_size	) % weight_channel);
		const int i_row = ((idx /= weight_channel	) % n_size);
		const int i_col = ((idx /= n_size	) % n_size);
		int input_row, input_col;

		// corresponding position of the input matrix and size of output matrix
		if (SAME){ // SAME padding scheme implemented
			input_row = i_kernel_row + i_row - padding;
			input_col = i_kernel_col + i_col - padding;
		}
		else{
			input_row = i_kernel_row + i_row;
			input_col = i_kernel_col + i_col;
		}
		if(input_row >= 0 && input_row < size && input_col >=0 && input_col < size){
			atomicAdd(&output[((i_channel % out_channel) * n_size + i_col) * n_size + i_row], 
						weight[(i_channel * kernel_size + i_kernel_col) * kernel_size + i_kernel_row] 
						* input[((i_channel % in_channel) * size + input_col) * size + input_row]);
		}
	}
}

/**name: fp_bias_conv
 * function: add bias to matrix after convolution operation
 * @param preact     input feature matrix after convolution
 * @param bias       bias term for each channel
 * @param size       size of input feature matrix (size * size)
 * @param n_channel  number of channels of input feature matrix
 */
__global__ void fp_bias_conv(float* preact, float* bias, const int size, const int n_channel)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	const int N = n_channel * size * size;

	for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) {
		int idx = n;
		const int i_channel = ((idx /= 1	) % n_channel);
		const int i_row = ((idx /= n_channel	) % size);
		const int i_col = ((idx /= size	) % size);

		preact[(i_channel * size + i_col) * size + i_row] += bias[i_channel];
	}
}

/**name:fp_preact_fc
 * function: matrix multiplication part for full connected layer
 * @param input        input matrix
 * @param preact       output matrix after FC
 * @param weight       weight matrix needed to execute full connected operation
 * @param size         size of input feature map of each channel
 * @param in_channel   nubmer of channels of input feature matrix
 * @param out_channel  number of channels of output feature matrix (1 * 1 * out_channel)
 */
__global__ void fp_preact_fc(float* input, float* preact, float* weight,
							const int size, const int in_channel, const int out_channel)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	const int weight_channel = in_channel * out_channel;
	const int N = out_channel * in_channel * size * size;  // number of elements of weight matrix

	for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) {
		int idx = n;
		const int i_channel = ((idx /= 1	) % weight_channel);
		const int i_row = ((idx /= weight_channel	) % size);
		const int i_col = ((idx /= size	) % size);

		atomicAdd(&preact[i_channel % out_channel], weight[(i_channel * size + i_col) * size + i_row] * input[((i_channel % in_channel) * size + i_col) * size + i_row]);
	}
}

/**name:fp_bias_fc
 * function: add bias term to each channel of FC output
 * @param preact     feature matrix after FC
 * @param bias       bias term for each channel
 * @param n_channel  number of channels of feature matrix
 */
__global__ void fp_bias_fc(float *preact, float *bias, const int n_channel)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	const int N = n_channel;

	for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) {
		preact[idx] += bias[idx];
	}
}

/**name: bp_weight_fc
 * function: compute the gradient of weight matrix of FC layer
 * @param d_weight       output gradient of weight matrix
 * @param d_preact       input gradient of feature matrix after FC layer
 * @param p_output       previous output feature matrix before FC layer
 * @param size           size of previous output feature matrix
 * @Param in_channel     number of channels of previous output feature matrix
 # @param out_channel    number of channels of input feature matrix after FC layer
 */
__global__ void bp_weight_fc(float *d_weight, float *d_preact, float *p_output,
							const int size, const int in_channel, const int out_channel)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	const int N = out_channel * in_channel * size * size;
	const int weight_channel = out_channel * in_channel;

	for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) {
		int idx = n;
		const int i_channel = ((idx /= 1	) % weight_channel);
		const int i_row = ((idx /= weight_channel	) % size);
		const int i_col = ((idx /= size	) % size);

		d_weight[(i_channel * size + i_col) * size + i_row] = d_preact[i_channel % out_channel] * p_output[((i_channel % in_channel) * size + i_col) * size + i_row];
	}
}

/**name: bp_bias_fc
 * function: update the bias term of FC layer during backpropagation
 * @param bias         bias term
 * @oaram d_preact     gradient of feature matrix
 * @param n_channel    number of channels of both bias term and feature matrix
 */
__global__ void bp_bias_fc(float *bias, float *d_preact, const int n_channel)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	const int N = n_channel;

	for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) {
		bias[idx] += dt * d_preact[idx];  // update bias term
	}
}

/**name: bp_output_conv
 * function: backward pass for convolution layer, get the gradient of each element of output from the gradient of next layer
 * @param d_output         gradient of output data matrix of convolution operation
 * @param n_weight         weight matrix of next layer
 * @param nd_preact        gradient of next layer
 * @param kernel_size      the size of weight matrix
 * @param n_size           the size of feature matrix of next layer
 * @param size             the size of feature matrix of current layer
 * @param in_channel       the number of channels for input data matrix
 * @param out_channel      the number of channels for output data matrix
 * @param CONV             boolean indcating whether the next layer is a convolution layer
 * @Param SAME             boolean indicating whether "SAME" padding was used during forward pass of next layer
 */
 
/*
__global__ void bp_output_conv(float *d_output, float *n_weight, float *nd_preact, const int size,
							const int kernel_size, const int n_size, const int in_channel, const int out_channel, bool CONV, bool SAME)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	const int N = kernel_size * kernel_size * n_size * n_size * in_channel * out_channel;
	const int weight_channel = out_channel * in_channel;
	const int padding = (kernel_size - 1) / 2;   // must be int
	int input_row, input_col;

	for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) { 
		int idx = n;
		if (CONV){   // the next layer is convolution or maxpooling
			const int i_channel = ((idx /= 1	) % weight_channel);
			const int i_kernel_row = ((idx /= weight_channel) % kernel_size); 
			const int i_kernel_col = ((idx /= kernel_size) % kernel_size);
			const int i_row = ((idx /= kernel_size	) % n_size);
			const int i_col = ((idx /= n_size) % n_size);
			
			if(SAME){     // with padding situation
				input_row = i_row + i_kernel_row - padding;
				input_col = i_col + i_kernel_col - padding;
			}
			else{
				input_row = i_row + i_kernel_row;
				input_col = i_col + i_kernel_col;
			}
			if(input_row >= 0 && input_row < size && input_col >=0 && input_col < size){
				atomicAdd(&d_output[((i_channel % in_channel) * size + input_col) * size + input_row], 
							n_weight[(i_channel * kernel_size + i_kernel_col) * kernel_size + i_kernel_row] 
							* nd_preact[((i_channel % out_channel) * n_size + i_col) * n_size + i_row]);
			}
		}
		else{
			const int i_channel = ((idx /= 1) % weight_channel);
			const int i_row = ((idx /= weight_channel) % size);
			const int i_col = ((idx /= size	) % size);
	
			atomicAdd(&d_output[((i_channel % in_channel) * size + i_col) * size + i_row], 
			nd_preact[i_channel % out_channel] * n_weight[(i_channel * size + i_col) * size + i_row]);
	
		}
	}
}
*/
// test vision for output conv

__global__ void bp_output_conv(float *d_output, float *weight, float *nd_preact, const int size,
							const int kernel_size, const int n_size, const int in_channel, const int out_channel, bool CONV, bool SAME)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	const int N = kernel_size * kernel_size * size * size * in_channel * out_channel;
	const int weight_channel = out_channel * in_channel;
	const int padding = (kernel_size - 1) / 2;   // must be int

	for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) { 
		int idx = n;
    int bpinput_row;
    int bpinput_col;
		const int i_channel = ((idx /= 1	) % weight_channel);
		const int i_kernel_row = ((idx /= weight_channel) % kernel_size); 
		const int i_kernel_col = ((idx /= kernel_size) % kernel_size);
		const int i_row = ((idx /= kernel_size	) % size);
		const int i_col = ((idx /= size) % size);   

		if (SAME){ // SAME padding scheme implemented
			bpinput_row = i_kernel_row + i_row - padding;
			bpinput_col = i_kernel_col + i_col - padding;
		}
		else{
			bpinput_row = i_kernel_row + i_row - 2 * padding;
			bpinput_col = i_kernel_col + i_col - 2 * padding;
		}
   
		if(bpinput_row >= 0 && bpinput_row < size && bpinput_col >=0 && bpinput_col < size){   
				atomicAdd(&d_output[((i_channel % in_channel) * size + i_col) * size + i_row], 
							weight[(i_channel * kernel_size + (kernel_size - i_kernel_col)) * kernel_size + kernel_size - i_kernel_row] 
							* nd_preact[((i_channel % out_channel) * n_size + bpinput_col) * n_size + bpinput_row]);
		}   
  }  
} 





/**name:bp_preact_conv
 * function: compute the gradient of current layer for update of weights and bias
 * @param d_preact  gradient of current layer
 * @param d_output  gradient of output of current layer
 * @param preact    feature matrix of current layer
 */
__global__ void bp_preact_conv(float *d_preact, float *d_output, float *preact, const int size, const int n_channel)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	const int N = n_channel * size * size;

	for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) {
		int idx = n;
		const int i_channel = ((idx /= 1	) % n_channel);
		const int i_col = ((idx /= n_channel	) % size);
		const int i_row = ((idx /= n_channel	) % size);

		const float o = step_function(preact[(i_channel * size + i_col) * size + i_row]);

		d_preact[(i_channel * size + i_col) * size + i_row] = d_output[(i_channel * size + i_col) * size + i_row] * o * (1 - o);
	}
}
/**name: bp_weight_conv
 * function: get the gradient of weight of convolution layer
 * @param d_weight       gradient of weight matrix
 * @param d_preact       gradient of feature matrix
 * @param p_output       previous output feature matrix
 * @param kernel_size    size of weight matrix
 * @param size           size of previous output feature matrix
 * @param n_size         size of feature matrix
 * @param in_channel     number of channels of previous output feature matrix
 * @param out_channel    number of channels of feature matrix
 * @param SAME           boolean indicating whether "SAME" padding is used in this convolution layer
 */
__global__ void bp_weight_conv(float* d_weight, float* d_preact, float* p_output, const int kernel_size, 
								const int size, const int n_size, const int in_channel, const int out_channel, bool SAME)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;
	const int N = kernel_size * kernel_size * n_size * n_size * in_channel * out_channel;  // total number of connections in this convolution
	const int weight_channel = in_channel * out_channel;  // actual number of channels of weight matrix
	const int padding = (kernel_size - 1) / 2;  // number of padding for both ends
	int input_row, input_col;

	// distribute certain number of connections to each thread regardless of detailed position and shape
	for(int n = N * pos / totalPos; n < N * (pos+1) / totalPos; n++){
		int idx = n;
		const int i_kernel_row = ((idx /= 1	) % kernel_size);  
		const int i_kernel_col = ((idx /= kernel_size	) % kernel_size);
		const int i_channel = ((idx /= kernel_size	) % weight_channel);
		const int i_row = ((idx /= weight_channel	) % n_size);
		const int i_col = ((idx /= n_size	) % n_size);

		// corresponding position of the input matrix
		if (SAME){ // SAME padding scheme implemented
			input_row = i_kernel_row + i_row - padding;
			input_col = i_kernel_col + i_col - padding;
		}
		else{
			input_row = i_kernel_row + i_row;
			input_col = i_kernel_col + i_col;
		}
		if(input_row >= 0 && input_row < size && input_col >=0 && input_col < size){
			atomicAdd(&d_weight[(i_channel * kernel_size + i_kernel_col) * kernel_size + i_kernel_row], 
						d_preact[((i_channel % out_channel) * n_size + i_col) * n_size + i_row] * p_output[((i_channel % in_channel) * size + input_col) + input_row]);
		}
	}
}
/**name: bp_bias_conv
 * function: update the bias terms of convolution layer
 * @param bias       bias term of convolution layer
 * @param d_preact   gradient of feature matrix of convolution layer
 * @Param size       size of feature matrix
 * @param n_channel  number of channels of feature matrix
 */
__global__ void bp_bias_conv(float *bias, float *d_preact, const int size, const int n_channel)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;

	const int N = n_channel * size * size;
	//const float d = pow(24.0f, 2.0f);   // what is this d for?

	for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) {
		int idx = n;
		const int i_channel = ((idx /= 1	) % n_channel);
		const int i_col = ((idx /= n_channel) % size);
		const int i_row = ((idx /= size	) % size);

		atomicAdd(&bias[i_channel], dt * d_preact[(i_channel * size + i_col) * size + i_row]);
		//atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);  // what is this d for?
	}
}



__global__ void fp_maxpool(float* output, float* input, const int kernel_size, const int size, const int n_size, const int in_channel, bool SAME)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;
	const int N = kernel_size * kernel_size * n_size * n_size * in_channel;  // total number of connections in this convolution
	const int padding = (kernel_size - 1) / 2;  // number of padding for both ends
	int input_row, input_col;
	// distribute certain number of connections to each thread regardless of detailed position and shape
	for(int n = N * pos / totalPos; n < N * (pos+1) / totalPos; n++){
		int idx = n;
		const int i_kernel_row = ((idx /= 1	) % kernel_size);  
		const int i_kernel_col = ((idx /= kernel_size	) % kernel_size);
		const int i_channel = ((idx /= kernel_size	) % in_channel);
		const int i_row = ((idx /= in_channel	) % n_size);
		const int i_col = ((idx /= n_size	) % n_size);
    float maxidx = (float)-1;
		// corresponding position of the input matrix and size of output matrix
		if (SAME){ // SAME padding scheme implemented
			input_row = i_kernel_row + i_row - padding;
			input_col = i_kernel_col + i_col - padding;
		}
		else{
			input_row = i_kernel_row + i_row;
			input_col = i_kernel_col + i_col;
		}
		if(input_row >= 0 && input_row < size && input_col >=0 && input_col < size){
			  if (input[((i_channel % in_channel) * size + input_col) * size + input_row] > maxidx)
            output[((i_channel % in_channel) * n_size + i_col) * n_size + i_row] = maxidx;
		}
	}
}


__global__ void bp_maxpool(float* d_preact, float* preact, float* p_output, const int kernel_size, const int size, const int n_size, const int in_channel, bool SAME)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;
	const int N = kernel_size * kernel_size * n_size * n_size * in_channel;  // total number of connections in this convolution
	const int padding = (kernel_size - 1) / 2;  // number of padding for both ends
	int input_row, input_col;

	// distribute certain number of connections to each thread regardless of detailed position and shape
	for(int n = N * pos / totalPos; n < N * (pos+1) / totalPos; n++){
		int idx = n;
		const int i_kernel_row = ((idx /= 1	) % kernel_size);  
		const int i_kernel_col = ((idx /= kernel_size	) % kernel_size);
		const int i_channel = ((idx /= kernel_size	) % in_channel);
		const int i_row = ((idx /= in_channel	) % n_size);
		const int i_col = ((idx /= n_size	) % n_size);
    float maxidx = (float)-1;
    idx = 0;
		// corresponding position of the input matrix
		if (SAME){ // SAME padding scheme implemented
			input_row = i_kernel_row + i_row - padding;
			input_col = i_kernel_col + i_col - padding;
		}
		else{
			input_row = i_kernel_row + i_row;
			input_col = i_kernel_col + i_col;
		}
		if(input_row >= 0 && input_row < size && input_col >=0 && input_col < size){
			  if (preact[((i_channel % in_channel) * size + input_col) * size + input_row] > maxidx)
          {
            preact[((i_channel % in_channel) * size + input_col) * size + input_row] = maxidx;
            idx = ((i_channel % in_channel) * size + input_col) * size + input_row;
          }
		}
    p_output[idx] = d_preact[((i_channel % in_channel) * n_size + i_col) * n_size + i_row];  
	}
}
