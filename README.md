# Solving with C using the FANN library

This project contains my attempt at using the MNIST handwritten dataset with pure C using the free open source neural network library "fann" to predict the numbers in the dataset.

## Goals

Benchmark the performance and my knowleadge of the fann library, as well as some minimal configurations for creating

## Environment

I will be using Linux (Debian) x64 to build and test the project, but it should not be heavily OS-dependant as everything is very straightforward.

## Building / Compilation

I'm a simple guy and this is a simple project so I will not be using tools that create instructions that create build steps that create executables.

So I will be compiling the project with the GNU Compiler Collection, which can be retrieved from the build-essentials package: `sudo apt install build-essential`.

`gcc -Wfatal-errors -Ifann/include -lm -o ./main main.c`

For documentation purposes, the version I'm running is `gcc (Debian 6.3.0-18+deb9u1) 6.3.0 20170516`.

## Credits

Huge thanks to the contributors to the tools and dataset used in this project as without them this project would not be possible.

 - [FANN](http://leenissen.dk/fann/wp/) Library is a free open source neural network library, which implements multilayer artificial neural networks in C with support for both fully connected and sparsely connected networks
 - [MNIST Handwritten Dataset](http://yann.lecun.com/exdb/mnist/)
