# A Neural Network in C to predict the output of the MNIST Handwritten dataset

This project contains my attempt at using the free open source neural network library "fann" to train a neural network to recognize the MNIST Handwritten dataset and predict the number depicted on the 28x28 grayscale image with an acceptable accuracy.

## Goals

Benchmark the performance and my knowleadge of the fann library, as well as some minimal configurations for experimentating with machine learning such as hyperparameter tuning.

## Environment

I will be using Linux (Debian 4.9) AMD64 to build and test the project, but it should not be heavily OS-dependant as everything is very straightforward.

## Building / Compilation

I'm a simple guy and this is a simple project so I will not be using tools that create instructions that create build steps that run compilers that create executables from my code because I think that is disproportionate.

I will be compiling the project with the GNU Compiler Collection, which can be retrieved from the build-essentials package: `sudo apt install build-essential`. The command I use to compile is the following:

```
gcc -Wfatal-errors -Ifann/include -lm -o ./main main.c
```

For documentation purposes, the version I'm running is `gcc (Debian 6.3.0-18+deb9u1) 6.3.0 20170516`.

## Results

Two variants were tested: Single network with 10 outputs and 10 networks with 1 output each:

### Version 1 - Monolithic (04/2020)

The code managed to load, train (100 epoch, 60000 train images) and validate it entirely in 11 minutes, without gcc optimization, in the smallest VM available at Google Compute (f1-micro, which is a single virtual CPU and 0.6 GB memory), achieving 54509 correct guesses out of 60000 (90.85 %) for training data and 9071 out of 10000 (90.71 %) for test data on a network with 4 fully-connected layers: 784 > 3 > 3 > 10, using incremental algorithm (weights are updated after each training set) and a learning rate of 0.1 or 0.9 depending on how you interpret that number.

By optimizing it with `-O3` at the compiler, I managed to get it to run in 2 minute and 52 seconds in the same environment, with 87.17 % accuracy for training data and 87.81 % for test data.

There are a lot of optimizations to be done, to improve accuracy by changing the network structure and hyper-parameters, or to improve the performance by removing costly stdout printing and its logic, but the goals of this project have been fully met.

A good command to compile and time the execution is this:

```
gcc -O3 -Wfatal-errors -Ifann/include -lm -o ./main main.c && time ./main
```

By changing the network layer sizes to 728, 49, 10, 10, and epochs to 15, I managed to get 95.67% accuracy after training for 5:39 (with -O3 optimization).

Future works could expand this project by the implementation of a Grid Search to find the optimum hyperparameter configuration. As described by the literature surrounding this dataset, pooling, max, flattening and deskeewing the input prior to feeding it to the network would yield vastly better results, although 'vastly' is debatable, since it would go up from 96% to 99%.

### Version 2 - Parallel (07/2020)

By observing that computing is advancing horizontally (more cpus) rather than vertically (faster cpu) nowadays, I developed a parallel version of this project where 10 networks are trained to recognize each a single digit, that is, a network receives a 28x28 image and outputs a single output with how probable that the digit is in the input image.

Note that there were a lot of changes in structure in this version, like FANN not being included in the compilation as a library but in the code as a single-file include and the addition of scripts to compile-and-execute in Windows (requires Visual Studio 2019) and Linux.

Obs: To get the best hyperparameters for the parallel version of the neural a random-search of 170 configurations was experimented with, that why there are 8 variants, they are the best out of the 170 configurations. Only 2 and 3 layer networks were tested and the best have a hidden layer size of 120 neurons and 81745 connections.

Each network that distinguish between a specific number and marks all others as non-matches has an accuracy of 96%. To infere a number from an image the 10 networks have to be fed with the input and the network with the highest value correspond to the resulting digit.

The best network experimented with delivered an accuracy of 96.17% (with low variance, something like 0.0001) out of the 1000 test dataset pairs (not used for training). The network was trained in 20 steps of 400 epoches each and 800 randomly-selected training dataset-pairs (for each of the 10 networks), this result is almost insignificantly above the past score of the monolitic version.

You may pass a number between 0 and 7 (inclusive) to the network to train the different variants, althought i ordered them so that 0 is the best and 7 the 8th best and, if the `./output` folder exists, it will write the network and its configuration in the FANN internal format (interpretable text file loaded with `fann_create_from_file`).

In conclusion the network can now stop if it reaches a high number of matching likehood (e.g. if the inference of digit 3 yields 90% certainty you can be pretty sure all others will be close to zero and stop the inference) or even process all digits in parallel, which should easily speed up the inference by a factor of 5, up to 10 times since the inference can be done in a 100% parallel fashion.

### Version 3 - Parallel with connection degradation (07/2020)

Same thing as parallel version but now each network starts degradating as it trains. Degrading is the process of removing networks with lowest weights as the networks trains. This removes useless multiplications of very low numbers, which have very low correlation with the output.

Each network, in addition to calculating its performance as it trains, outputs a csv (that can be oppened with libreoffice, google sheets or excel) to allow you to analyze the decrease in performance caused by the degradation.

Each network managed to keep a performance of 95% even when degradated by 90%. Probably because of some overfitting since the evaluation of the testing dataset when the network is degradated by 97.5% (~79k connections removed, ~2k left per digit), yields a performance of 74.21%.

x * 0.775 = 633530
x = 633530 / 0.775

Degrading the network by 84.998% removes a total of 694820 inference multiplications (leaving 122630 out of 817450) and yields a performance of 95.54% (guesses 9554 correctly and 446 incorrectly out of 10000 from the testing dataset). A second execution to confirm the performance resulted in 96.64% (336 incorrect).

Found it pretty interesting.

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

```

Neat, isnt it? Feel free to copy/edit it if you fancy it. It's in the code as `void print_grayscale_image(double * image, uint32_t width, uint32_t height)`

## Modified FANN Library

FANN library (described in credits) is included in this repository as a single-file library (in two options: `doublefann.h` and `floatfann.h`), that is not the original FANN library but a modified version changed by myself to include some activation functions and, obviously, compiled to a single file for convenience.

## Credits

Huge thanks to the contributors to the tools and dataset used in this project as without them this project would not be possible.

 - [FANN](http://leenissen.dk/fann/wp/) Library is a free open source neural network library, which implements multilayer artificial neural networks in C with support for both fully connected and sparsely connected networks. [Github Repository](https://github.com/libfann/fann/).
 - [MNIST Handwritten Dataset](http://yann.lecun.com/exdb/mnist/) is a database of of handwritten digits made from the larger subset NIST, it is good for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

## License

I should not be liable by any damage this code causes as it might be unreliable and untested.

Each dependency might have its own license statement and should be read accordingly.
