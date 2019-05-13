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
static Layer l_conv1 = Layer(3, 28, 24, 1, 8);

static Layer l_conv2 = Layer(1, 24, 24, 8, 8);
static Layer l_conv3 = Layer(3, 24, 24, 8, 8);
static Layer l_conv4 = Layer(5, 24, 24, 8, 8);
static Layer l_maxpool = Layer(3, 24, 24, 8, 8);

static Layer l_conv5 = Layer(1, 24, 24, 8, 8);
static Layer l_conv6 = Layer(1, 24, 24, 8, 8);
static Layer l_conv7 = Layer(1, 24, 24, 8, 8);

static Layer l_FC = Layer(24, 24, 1, 32, 10);

static float* concat_matrix, slice_1, slice_2, slice_3, slice_4, sum_matrix;



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

	cudaMalloc((void **)&concat_matrix, sizeof(float) * 24 * 24 * 32);
	cudaMalloc((void **)&slice_1, sizeof(float) * 24 * 24 * 8);
	cudaMalloc((void **)&slice_2, sizeof(float) * 24 * 24 * 8);
	cudaMalloc((void **)&slice_3, sizeof(float) * 24 * 24 * 8);
	cudaMalloc((void **)&slice_4, sizeof(float) * 24 * 24 * 8);
	cudaMalloc((void **)&sum_matrix, sizeof(float) * 24 * 24 * 8);
      
	loaddata();

        //test();
	learn();
	test();

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];
        //fprintf(stdout, "%f\n", data[14][14]);

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_conv1.clear();
	l_conv2.clear();
	l_conv3.clear();
	l_conv4.clear();
	l_conv5.clear();
	l_conv6.clear();
	l_conv7.clear();
	l_maxpool.clear();
	l_FC.clear();

	clock_t start, end;
	start = clock();

	l_input.setOutput((float *)input);
	//l_input.Out();
	// Conv1
	fp_conv<<<64, 64>>>(l_conv1.preact, l_input.output, l_conv1.weight, l_conv1.kernel_size, 
						l_conv1.in_size, l_conv1.out_size, l_conv1.in_channel, l_conv1.out_channel, false);

	fp_bias_conv<<<64, 64>>>(l_conv1.preact, l_conv1.bias, l_conv1.out_size, l_conv1.out_channel);
	apply_step_function<<<64, 64>>>(l_conv1.preact, l_conv1.output, l_conv1.out_size * l_conv1.out_size * l_conv1.out_channel);
	
	
	// parallel block
	fp_four_parallel<<<4,1>>>(concat_matrix, l_conv2, l_conv3, l_conv4, l_maxpool, l_conv5, l_conv6, l_conv7, l_conv1);
	 
	// FC
	fp_preact_fc<<<64, 64>>>(concat_matrix, l_FC.preact, l_FC.weight, l_FC.in_size, l_FC.in_channel, l_FC.out_channel);
	fp_bias_fc<<<64, 64>>>(l_FC.preact, l_FC.bias, l_FC.out_channel);
	apply_step_function<<<64, 64>>>(l_FC.preact, l_FC.output, l_FC.out_size * l_FC.out_size * l_FC.out_channel);
	//l_FC.Out();
	
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
	bp_output_fc<<<64, 64>>>(l_FC.d_output, l_FC.d_preact, l_FC.weight, l_FC.in_size, l_FC.in_channel, l_FC.out_channel);
	//l_FC.dOut();

	// decat
	decat<<<64,64>>>(l_FC.d_output, &slice_1, &slice_2, &slice_3, &slice_4,
		l_FC.in_size, l_conv2.out_channel, l_conv3.out_channel, l_conv4.out_channel, l_maxpool.out_channel);

	// parallel block
	bp_four_parallel<<<4,1>>>(&sum_matrix, l_conv2, l_conv3, l_conv4, l_maxpool, l_conv5, l_conv6, l_conv7, &slice_1, &slice_2, &slice_3, &slice_4, l_conv1.output);
 

	// Conv1
	//bp_output_conv<<<64, 64>>>(l_conv1.d_output, l_conv1.weight, l_conv2.d_preact, l_conv1.in_size, l_conv2.kernel_size, 
	bp_output_conv<<<64, 64>>>(l_conv1.d_output, l_conv1.weight, &sum_matrix, l_conv1.in_size, l_conv2.kernel_size, 
								l_conv2.out_size, l_conv2.in_channel, l_conv2.out_channel, true, true);
	bp_preact_conv<<<64, 64>>>(l_conv1.d_preact, l_conv1.d_output, l_conv1.preact, l_conv1.out_size, l_conv1.out_channel);
	bp_weight_conv<<<64, 64>>>(l_conv1.d_weight, l_conv1.d_preact, l_conv1.output, l_conv1.kernel_size, l_conv1.in_size,
		l_conv1.out_size, l_conv1.in_channel, l_conv1.out_channel, false);
	bp_bias_conv<<<64, 64>>>(l_conv1.bias, l_conv1.d_preact, l_conv1.out_size, l_conv1.out_channel);
	//l_conv1.dOut();


	apply_grad<<<64, 64>>>(l_FC.weight, l_FC.d_weight, l_FC.M * l_FC.N);
	apply_grad<<<64, 64>>>(l_conv1.weight, l_conv1.d_weight, l_conv1.M * l_conv1.N);
	apply_grad<<<64, 64>>>(l_conv2.weight, l_conv2.d_weight, l_conv2.M * l_conv2.N);
	apply_grad<<<64, 64>>>(l_conv3.weight, l_conv3.d_weight, l_conv3.M * l_conv3.N);
	apply_grad<<<64, 64>>>(l_conv4.weight, l_conv4.d_weight, l_conv4.M * l_conv4.N);
	apply_grad<<<64, 64>>>(l_conv5.weight, l_conv5.d_weight, l_conv5.M * l_conv5.N);
	apply_grad<<<64, 64>>>(l_conv6.weight, l_conv6.d_weight, l_conv6.M * l_conv6.N);
	apply_grad<<<64, 64>>>(l_conv7.weight, l_conv7.d_weight, l_conv7.M * l_conv7.N);

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

		for (int i = 0; i < 4; ++i) {
			float tmp_err;
			int index = rand() % train_cnt;
			time_taken += forward_pass(train_set[index].data);
			
			l_FC.bp_clear();
			l_maxpool.bp_clear();
			l_conv1.bp_clear();
			l_conv2.bp_clear();
			l_conv3.bp_clear();
			l_conv4.bp_clear();
			l_conv5.bp_clear();
			l_conv6.bp_clear();
			l_conv7.bp_clear();

			// Euclid distance of train_set[i]
			//l_FC.Out();
			calcLoss<<<10, 1>>>(l_FC.d_preact, l_FC.output, train_set[index].label, 10);
			//l_FC.dOut();
			//cudaMemcpy(fuck, l_FC.d_preact, sizeof(float) * 10, cudaMemcpyDeviceToHost);
			//for(int i = 0; i < 10; i++){
			// 	fprintf(stdout, " %f ", fuck[i]);
			// }                        
			// fprintf(stdout, "\n");
			cublasSnrm2(blas, 10, l_FC.d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken += back_pass();
		}
		//printf("jfhgodsufg\n");
		err /= 4;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);
		//l_FC.Out();
		if (err < 0) {   // threshold
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
