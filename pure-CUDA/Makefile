all:
	/usr/local/cuda-10.0/bin/nvcc -lcuda -lcublas -lcudadevrt -lineinfo layer.cu sequential.cu -o Seq  -arch=compute_35 -rdc=true -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.0/bin/nvcc -lcuda -lcublas -lcudadevrt -lineinfo layer.cu parallel.cu -o Para  -arch=compute_35 -rdc=true -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.0/bin/nvcc -lcuda -lcublas -lineinfo layer_demo.cu main.cu -o main  -arch=compute_35 -Wno-deprecated-gpu-targets

run:
	./Seq
clean:
	rm Seq Para
