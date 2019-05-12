#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28, 0, 1);
static Layer l_conv1 = Layer(5, 28, 24, 1, 8);
static Layer l_conv2 = Layer(1, 24, 24, 8, 16);
static Layer l_maxpool = Layer(1, 24, 24, 16, 16);
static Layer l_FC = Layer(24, 24, 1, 16, 10);
// static Layer l_input = Layer(0, 0, 28*28);
// static Layer l_c1 = Layer(5*5, 6, 24*24*6);
// static Layer l_s1 = Layer(4*4, 1, 6*6*6);
// static Layer l_f = Layer(6*6*6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}

	loaddata();
	learn();
	test();

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	//float input[28][28];
  float *input; //= (float *)malloc(28 * 28 * sizeof(float));
  cudaMallocHost(&input, sizeof(float) * 28 * 28);
	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i * 28 + j] = data[i][j]; 
		}
	}

	l_input.clear();
	l_conv1.clear(); 
	l_conv2.clear();
	l_maxpool.clear(); 
	l_FC.clear();


	clock_t start, end;
	start = clock();
  fprintf(stdout, "%f\n", input[0] );	

	l_input.setOutput(input);
  fprintf(stdout, "after setoutput\n");	
	// Conv1
	fp_conv<<<64, 64>>>(l_conv1.preact, l_input.output, l_conv1.weight, l_conv1.kernel_size, 
						l_conv1.in_size, l_conv1.out_size, l_conv1.in_channel, l_conv1.out_channel, false);
                                               
  fprintf(stdout, "%f\n", l_conv1.preact[0]);
	fp_bias_conv<<<64, 64>>>(l_conv1.preact, l_conv1.bias, l_conv1.out_size, l_conv1.out_channel);
	apply_step_function<<<64, 64>>>(l_conv1.preact, l_conv1.output, l_conv1.out_size * l_conv1.out_size * l_conv1.out_channel);

	// Conv2
	fp_conv<<<64, 64>>>(l_conv2.preact, l_conv1.output, l_conv2.weight, l_conv2.kernel_size, 
			l_conv2.in_size, l_conv2.out_size, l_conv2.in_channel, l_conv2.out_channel, true);
	fp_bias_conv<<<64, 64>>>(l_conv2.preact, l_conv2.bias, l_conv2.out_size, l_conv2.out_channel);
	apply_step_function<<<64, 64>>>(l_conv2.preact, l_conv2.output, l_conv2.out_size * l_conv2.out_size * l_conv2.out_channel);

	// Maxpooling
	fp_maxpool<<<64, 64>>>(l_maxpool.output, l_conv2.output, l_maxpool.kernel_size, l_maxpool.in_size, l_maxpool.out_size, l_maxpool.out_channel, true);

	// FC
	fp_preact_fc<<<64, 64>>>(l_maxpool.output, l_FC.preact, l_FC.weight, l_FC.in_size, l_FC.in_channel, l_FC.out_channel);
	fp_bias_fc<<<64, 64>>>(l_FC.preact, l_FC.bias, l_FC.out_channel);
	apply_step_function<<<64, 64>>>(l_FC.preact, l_FC.output, l_FC.out_size * l_FC.out_size * l_FC.out_channel);
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;
	start = clock();
	// FC
	bp_weight_fc<<<64, 64>>>(l_FC.d_weight, l_FC.d_preact, l_maxpool.output, l_FC.in_size, l_FC.in_channel, l_FC.out_channel);
	bp_bias_fc<<<64, 64>>>(l_FC.bias, l_FC.d_preact, l_FC.out_channel);

	// Maxpooling
	bp_maxpool<<<64, 64>>>(l_maxpool.d_preact, l_maxpool.output, l_conv2.output, l_maxpool.kernel_size, l_maxpool.in_size, l_maxpool.out_size, l_maxpool.out_channel, true);

	// Conv2
	bp_output_conv<<<64, 64>>>(l_conv2.d_output, l_conv2.weight, l_maxpool.d_preact, l_conv2.in_size, l_maxpool.kernel_size, 
								l_maxpool.out_size, l_maxpool.in_channel, l_maxpool.out_channel, true, true);
	bp_preact_conv<<<64, 64>>>(l_conv2.d_preact, l_conv2.d_output, l_conv2.preact, l_conv2.out_size, l_conv2.out_channel);
	bp_weight_conv<<<64, 64>>>(l_conv2.d_weight, l_conv2.d_preact, l_conv1.output, l_conv2.kernel_size, l_conv2.in_size,
								l_conv2.out_size, l_conv2.in_channel, l_conv2.out_channel, true);
	bp_bias_conv<<<64, 64>>>(l_conv2.bias, l_conv2.d_preact, l_conv2.out_size, l_conv2.out_channel);

	// Conv1
	bp_output_conv<<<64, 64>>>(l_conv1.d_output, l_conv1.weight, l_conv2.d_preact, l_conv1.in_size, l_conv2.kernel_size, 
								l_conv2.out_size, l_conv2.in_channel, l_conv2.out_channel, true, true);
	bp_preact_conv<<<64, 64>>>(l_conv1.d_preact, l_conv1.d_output, l_conv1.preact, l_conv1.out_size, l_conv1.out_channel);
	bp_weight_conv<<<64, 64>>>(l_conv1.d_weight, l_conv1.d_preact, l_conv1.output, l_conv1.kernel_size, l_conv1.in_size,
		l_conv1.out_size, l_conv1.in_channel, l_conv1.out_channel, false);
	bp_bias_conv<<<64, 64>>>(l_conv1.bias, l_conv1.d_preact, l_conv1.out_size, l_conv1.out_channel);


	apply_grad<<<64, 64>>>(l_FC.weight, l_FC.d_weight, l_FC.M * l_FC.N);
	apply_grad<<<64, 64>>>(l_conv1.weight, l_conv1.d_weight, l_conv1.M * l_conv1.N);
	apply_grad<<<64, 64>>>(l_conv2.weight, l_conv2.d_weight, l_conv2.M * l_conv2.N);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

static void learn()
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float err;
	int iter = 1000;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			time_taken += forward_pass(train_set[i].data);

			l_FC.bp_clear();
			l_maxpool.bp_clear();
			l_conv2.bp_clear();
			l_conv1.bp_clear();
      
			// Euclid distance of train_set[i]
			calcLoss<<<10, 1>>>(l_FC.d_preact, l_FC.output, train_set[i].label, 10);
			for(int j = 0; j < 10; j++){
			  fprintf(stdout, "%f, %d\n", l_FC.output[j], train_set[i].label);
			}
			cublasSnrm2(blas, 10, l_FC.d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken += back_pass();
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Time - %lf\n", time_taken);
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_FC.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}
