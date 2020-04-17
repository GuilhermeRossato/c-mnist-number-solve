
#pragma once

/**
 * Struct: hyperparameters
 *
 * Holds hyper parameters used for the creation of a neural network
 * Should be created with `create_hyperparameters` and destroyed with `destroy_hyperparameters`
*/
struct hyperparameters;

/**
 * Function: create_hyperparameters
 *
 * Allocates a space in memory for a hyperparameter, populate and return it.
 *
 * Uses parameters to variate the network format
*/
struct hyperparameters * create_hyperparameters(int input_x, int input_y, int output_size);

/**
 * Function: destroy_hyperparameters
 *
 * Deallocates a hyperparameter object.
*/
void destroy_hyperparameters(struct hyperparameters * h);

/**
 * Function: create_network_with_hyperparameters
 *
 * Translates hyperparameters struct into the necessary inputs to create
 * the neural network, and returns the created new neural network.
 *
 * As per instructions on the fann library, the created neural network must be
 * destroyed explicitly by calling `fann_destroy`.
*/
struct fann * create_network_with_hyperparameters(struct hyperparameters * h);

/**
 * Function: print_hyperparameters
 *
 * Prints the hyperparameters with printf (standard output)
*/
void print_hyperparameters(struct hyperparameters * h);