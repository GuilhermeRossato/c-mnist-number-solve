#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define FANN_NO_DLL
#define FANN_NO_SEED
#define DISABLE_PARALLEL_FANN

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
    h->training_algorithm = TRAINING_ALGORITHM_INCREMENTAL;
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
struct fann_train_data * fann_create_train_pointer_array(unsigned int num_data, unsigned int num_input, double **input, unsigned int num_output, double **output);
void fann_train_on_file(struct fann *ann, const char *filename, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error);

int main() {
    srand((unsigned int) time(0));

    struct hyperparameters * h = create_hyperparameters(0);

    printf("Creating network.\n");
    struct fann * ann = create_network_with_hyperparameters(h);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    printf("Creating data.\n");
    double ** input = malloc(sizeof(double*) * 4);
    double ** output = malloc(sizeof(double*) * 4);
    for (int i = 0; i < 4; i++) {
        input[i] = malloc(sizeof(double) * 2);
        output[i] = malloc(sizeof(double) * 1);
        int x = i % 2;
        int y = i / 2;

        input[i][0] = x == 0 ? -1.0 : 1.0;
        input[i][1] = y == 0 ? -1.0 : 1.0;
        output[i][0] = x != y ? 1.0 : -1.0;
    }

    struct fann_train_data * data = fann_create_train_pointer_array(
        4, // Amount of training data
        2, // Input size
        input,
        1, // Output size
        output
    );

    if (rand()%2) {
        printf("Training network on data struct:\n");

        fann_train_on_data(
            ann,
            data,
            50000, // max epochs
            5000, // epochs between reports
            0.001 // desired error
        );
    } else {
        printf("Training network on file 'xor.data':\n");

        fann_train_on_file(
            ann,
            "xor.data",
            50000, // max epochs
            10000, // epochs between reports
            0.001 // desired error
        );
    }

    printf("Testing network:\n");
    for (int i = 0; i < 4; i++) {
        double * result = fann_run(ann, input[i]);
        double * expected = output[i];
        printf("%5.1f, %5.1f = %6.2f (expected %6.2f, error: %6.2f)\n", input[i][0], input[i][1], result[0], expected[0], expected[0] - result[0]);
    }

    for (int i = 0; i < 4; i++) {
        free(input[i]);
        free(output[i]);
    }
    free(input);
    free(output);
    fann_destroy_train(data);

    fann_destroy(ann);
    destroy_hyperparameters(h);

    return 0;
}