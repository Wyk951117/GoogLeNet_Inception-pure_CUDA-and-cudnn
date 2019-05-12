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

__global__ void fp_maxpool(float* output, float* input, const int kernel_size, const int size, const int n_size, const int in_channel, bool SAME)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalPos = blockDim.x * gridDim.x;
	const int N = kernel_size * kernel_size * n_size * n_size * in_channel;  // total number of connections in this convolution
	const int padding = (kernel_size - 1) / 2;  // number of padding for both ends

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
			const int input_row = i_kernel_row + i_row - padding;
			const int input_col = i_kernel_col + i_col - padding;
		}
		else{
			const int input_row = i_kernel_row + i_row;
			const int input_col = i_kernel_col + i_col;
		}
		if(input_row >= 0 && input_col < size && input_col >=0 && input_col < size){
			  if (input[((i_channel % in_channel) * size + input_col) * size + input_row]) > maxidx)
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

	// distribute certain number of connections to each thread regardless of detailed position and shape
	for(int n = N * pos / totalPos; n < N * (pos+1) / totalPos; n++){
		int idx = n;
		const int i_kernel_row = ((idx /= 1	) % kernel_size);  
		const int i_kernel_col = ((idx /= kernel_size	) % kernel_size);
		const int i_channel = ((idx /= kernel_size	) % in_channel);
		const int i_row = ((idx /= weight_channel	) % n_size);
		const int i_col = ((idx /= n_size	) % n_size);
    float maxidx = (float)-1;
    int idx = 0;
		// corresponding position of the input matrix
		if (SAME){ // SAME padding scheme implemented
			const int input_row = i_kernel_row + i_row - padding;
			const int input_col = i_kernel_col + i_col - padding;
		}
		else{
			const int input_row = i_kernel_row + i_row;
			const int input_col = i_kernel_col + i_col;
		}
		if(input_row >= 0 && input_col < size && input_col >=0 && input_col < size){
			  if (preact[((i_channel % in_channel) * size + input_col) * size + input_row] > maxidx)
          {
            preact[((i_channel % in_channel) * size + input_col) * size + input_row] = maxidx;
            idx = ((i_channel % in_channel) * size + input_col) * size + input_row;
          }
		}
    p_output[idx] = d_preact[((i_channel % in_channel) * n_size + i_col) * n_size + i_row];  
	}
}