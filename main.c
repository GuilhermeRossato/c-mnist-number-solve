#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define FANN_NO_DLL
#define FANN_NO_SEED

#include "fann/doublefann.c"
#include "fann/include/fann.h"

/**
 * Struct: hyperparameters
 *
 * Holds hyper parameters used for the creation of a neural network
 * Should be created with `create_hyperparameters` and destroyed with `destroy_hyperparameters`
 */
struct hyperparameters {
    unsigned int layers_size;
    unsigned int * layers;
    unsigned int is_random_weight;
    enum {
        TRAINING_ALGORITHM_INCREMENTAL = 0, // Weights are updated after each training set
        TRAINING_ALGORITHM_BATCH, // Standard backpropagation algorithm
	    TRAINING_ALGORITHM_RPROP, // The iRPROP training algorithm which is described by [Igel and Husken, 2000]
	    TRAINING_ALGORITHM_QUICKPROP, // The quickprop training algorithm is described by [Fahlman, 1988]
        TRAINING_ALGORITHM_SARPROP // I have no idea what this is
    } training_algorithm;
};

/**
 * Function: create_hyperparameters
 *
 * Allocates a space in memory for a hyperparameter, populate and return it.
*/
struct hyperparameters * create_hyperparameters(int processId) {
    struct hyperparameters * h = malloc(sizeof(struct hyperparameters));
    h->layers_size = 4;
    h->layers = malloc(sizeof(unsigned int) * (h->layers_size + 1));
    h->layers[0] = 2;
    h->layers[1] = 8;
    h->layers[2] = 8;
    h->layers[3] = 1;
    h->layers[4] = 0;
    h->is_random_weight = 1;
    h->training_algorithm = FANN_TRAIN_BATCH;
    return h;
}

/**
 * Function: destroy_hyperparameters
 *
 * Deallocates a hyperparameter object.
*/
void destroy_hyperparameters(struct hyperparameters * h) {
    free(h->layers);
    free(h);
}

struct fann * fann_create_standard_array(unsigned int num_layers, const unsigned int * layers);

/**
 * Function: create_network_with_hyperparameters
 *
 * Translates hyperparameters struct into the necessary inputs to create
 * the neural network, and returns the created new neural network.
 *
 * As per instructions on the fann library, the created neural network must be
 * destroyed explicitly by calling `fann_destroy`.
*/
struct fann * create_network_with_hyperparameters(struct hyperparameters * h) {
    struct fann * ann = fann_create_standard_array(h->layers_size, h->layers);

    fann_set_training_algorithm(ann, h->training_algorithm);

    if (h->is_random_weight) {
        fann_randomize_weights(ann, -0.1, 0.1);
    } else {
        // Uses Widrow + Nguyen's algorithm, but that requires data.
        // fann_init_weight(ann, data);
    }

    return ann;
}

void fann_randomize_weights(struct fann * ann, double min_weight, double max_weight);

int main() {
    srand((unsigned int) time(0));

    struct hyperparameters * h = create_hyperparameters(0);

    struct fann * ann = create_network_with_hyperparameters(h);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    fann_print_connections(ann);

    fann_destroy(ann);
    destroy_hyperparameters(h);

    return 0;
}