# Makefile
# Purpose: Direct compilation wrapper for the restructured modular educational micrograd library.

.PHONY: all build test run-demo clean

all: test run-demo

build: test_micrograd demo

test_micrograd: test/test_micrograd.cpp micrograd.hpp autograd/Value.hpp autograd/Value.cpp neural_network/NN.hpp neural_network/NN.cpp
	clang++ -O3 -std=c++17 test/test_micrograd.cpp autograd/Value.cpp neural_network/NN.cpp -I. -o test_micrograd

demo: demo.cpp micrograd.hpp autograd/Value.hpp autograd/Value.cpp neural_network/NN.hpp neural_network/NN.cpp training/training_loop.hpp training/training_loop.cpp
	clang++ -O3 -std=c++17 demo.cpp autograd/Value.cpp neural_network/NN.cpp training/training_loop.cpp -I. -o demo

test: test_micrograd
	./test_micrograd

run-demo: demo
	./demo

clean:
	rm -f test_micrograd demo
