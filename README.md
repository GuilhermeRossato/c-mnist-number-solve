# Solving with C using the FANN library

This project contains my attempt at using the MNIST handwritten dataset with pure C using the free open source neural network library "fann" to predict the numbers in the dataset.

## Goals

Benchmark the performance and my knowleadge of the fann library, as well as some minimal configurations for experimentating with machine learning such as hyperparameter tuning.

## Environment

I will be using Linux (Debian 4.9) AMD64 to build and test the project, but it should not be heavily OS-dependant as everything is very straightforward.

## Building / Compilation

I'm a simple guy and this is a simple project so I will not be using tools that create instructions that create build steps that create executables.

So I will be compiling the project with the GNU Compiler Collection, which can be retrieved from the build-essentials package: `sudo apt install build-essential`.

```
gcc -Wfatal-errors -Ifann/include -lm -o ./main main.c
```

For documentation purposes, the version I'm running is `gcc (Debian 6.3.0-18+deb9u1) 6.3.0 20170516`.

## Results

The code managed to load, train (100 epoch, 60000 train images) and validate it entirely in 11 minutes, without gcc optimization, in the smallest VM available at Google Compute (f1-micro, which is a single virtual CPU and 0.6 GB memory), achieving 54509 correct guesses out of 60000 (90.85 %) for training data and 9071 out of 10000 (90.71 %) for test data on a network with 4 fully-connected layers: 784 > 3 > 3 > 10, using incremental algorithm (weights are updated after each training set) and a learning rate of 0.1 or 0.9 depending on how you interpret that number.

By optimizing it with `-O3` at the compiler, I managed to get it to run in 2 minute and 52 seconds in the same environment, with 87.17 % accuracy for training data and 87.81 % for test data.

There are a lot of optimizations to be done, to improve accuracy by changing the network structure and hyper-parameters, or to improve the performance by removing costly stdout printing and its logic, but the goals of this project have been fully met.

A good command to compile and time the execution is this:

```
gcc -O3 -Wfatal-errors -Ifann/include -lm -o ./main main.c && time ./main
```

By changing the network layer sizes to 728, 49, 10, 10, and epochs to 15, I managed to get 95.67% accuracy after training for 5:39 (with -O3 optimization).

Future works could expand this project by the implementation of a Grid Search to find the optimum hyperparameter configuration. As described by the literature surrounding this dataset, pooling, max, flattening and deskeewing the input prior to feeding it to the network would yield vastly better results, although 'vastly' is debatable, since it would go up from 96% to 99%.

## Visualizing input

I managed to write a really nice function (`print_grayscale_image`) to print the number on the console, like so:
```

            raR@@lJar
           J#&@KR&&@S
         `c%&K?`'6&@t
         s&&4:  x&&D,
        ]@&Z:   X&&|
       1@g9-  .|Q@[
      :#&9   \b&&S'
      ^&&;^FE$@&&(
      :#&@&&&88&#
       r0@X3/^p&j
             L@@C
             z@&,
              @&,
             z@&,
             i@&,
             =@@,
              N&r
              s&K/
              :4&U'
               :Z&/

Real number: 9
Predicted: 9 (-0.14)
```

Neat, isnt it? Feel free to copy it if you fancy it.

## Credits

Huge thanks to the contributors to the tools and dataset used in this project as without them this project would not be possible.

 - [FANN](http://leenissen.dk/fann/wp/) Library is a free open source neural network library, which implements multilayer artificial neural networks in C with support for both fully connected and sparsely connected networks. [It also has a github repo.](https://github.com/libfann/fann/).
 - [MNIST Handwritten Dataset](http://yann.lecun.com/exdb/mnist/) is a database of of handwritten digits made from the larger subset NIST, it is good for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.
