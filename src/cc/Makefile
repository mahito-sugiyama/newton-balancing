INCLUDE	= -I/usr/local/include/eigen3

matrix: main_matrix.cc matrixBalancing.h
	g++ -O3 $(INCLUDE) -std=c++11 -march=native main_matrix.cc -o matrixbalance

tensor: main_tensor.cc tensorBalancing.h
	g++ -O3 $(INCLUDE) -std=c++11 -march=native main_tensor.cc -o tensorbalance
