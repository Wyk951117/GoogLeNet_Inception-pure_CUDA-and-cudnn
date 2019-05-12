all:
	/usr/local/cuda-10.0/bin/nvcc -lcuda -lcublas *.cu -o CNN  -arch=compute_35 -Wno-deprecated-gpu-targets

run:
	./CNN
clean:
	rm CNN
