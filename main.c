#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define FANN_NO_DLL
#define FANN_NO_SEED
#define DISABLE_PARALLEL_FANN

#include "fann/doublefann.h"

#include "idx_reader.h"
#include "idx_reader.c"

#define LOAD_NETWORKS 0
#define TRAIN_NETWORKS 1
#define SAVE_NETWORKS 1
#define IS_INPUT_ZERO_TO_ONE 1
#define IS_OUTPUT_ZERO_TO_ONE 1

int dataset_size;
int epoch_count;

struct fann * fann_create_standard_array(unsigned int num_layers, const unsigned int * layers);
void fann_randomize_weights(struct fann * ann, double min_weight, double max_weight);
struct fann_train_data * fann_create_train_pointer_array(unsigned int num_data, unsigned int num_input, double **input, unsigned int num_output, double **output);
void fann_train_on_file(struct fann *ann, const char *filename, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error);

enum source_type_t {source_type_test, source_type_train};
enum input_type_t {input_type_image, input_type_label};

struct idx_struct * create_idx_data_by_loading_file(
    enum source_type_t source_type,
    enum input_type_t input_type
) {
    char filename[256];
    snprintf(
        filename,
        256-1,
        "./data/%s-%s-ubyte",
        source_type == source_type_test ? "test" : "train",
        input_type == input_type_image ? "images.idx3" : "labels.idx1"
    );
    return create_idx_from_file(filename);
}

void destroy_idx_data(struct idx_struct * idx) {
    if (idx == NULL) {
        return;
    }
    free(idx->dimensions);
    free(idx->data);
    free(idx);
}

// Used by print_grayscale_image function
void print_grayscale_image(double * image, uint32_t width, uint32_t height) {
    const static char letters_by_occupancy[95] = {' ', '`', '.', '-', '\'', ':', '_', ',', '^', '"', '~', ';', '!', '\\', '>', '/', '=', '*', '<', '+', 'r', 'c', 'v', 'L', '?', ')', 'z', '{', '(', '|', 'T', '}', 'J', '7', 'x', 's', 'u', 'n', 'Y', 'i', 'C', 'y', 'l', 't', 'F', 'w', '1', 'o', '[', ']', 'f', '3', 'I', 'j', 'Z', 'a', 'e', '5', 'V', '2', 'h', 'k', 'S', 'U', 'q', '9', 'P', '6', '4', 'd', 'K', 'p', 'A', 'E', 'b', 'O', 'G', 'm', 'R', 'H', 'X', 'N', 'M', 'D', '8', 'W', '#', '0', 'B', '$', '%', 'Q', 'g', '&', '@'};

    uint32_t index;
    char letter;
    for (uint32_t y = 0; y < height; y++) {
        int is_line_empty = 1;
        for (uint32_t x = 0; x < width; x++) {
            if (image[x + y * width] != -1.0f) {
                is_line_empty = 0;
                break;
            }
        }
        if (y != 0 && y != height-1 && is_line_empty) {
            continue;
        }
        for (uint32_t x = 0; x < width; x++) {
            index = 95.0 * ((1.0 + image[x + y * width]) / 2.0);
            if (index < 0)
                index = 0;
            else if (index >= 95)
                index = 94;
            putchar(letters_by_occupancy[index]);
        }
        putchar('\n');
    }
}

int validate_data_from_files(struct idx_struct * train_images, struct idx_struct * train_labels, struct idx_struct * test_images, struct idx_struct * test_labels) {
    if (!train_images || !train_labels || !test_images || !test_labels) {
        printf("Closing due to file error\n");
        return 0;
    } else if (train_images->dimensions[0] != train_labels->dimensions[0]) {
        printf("Training image amount does not match label amount (%d != %d)\n", train_images->dimensions[0], train_labels->dimensions[0]);
        return 0;
    } else if (test_images->dimensions[0] != test_labels->dimensions[0]) {
        printf("Test image amount does not match label amount (%d != %d)\n", test_images->dimensions[0], test_labels->dimensions[0]);
        return 0;
    } else if (train_images->dimensions_size != 3 || train_labels->dimensions_size != 1 || test_images->dimensions_size != 3 || test_labels->dimensions_size != 1) {
        printf(
            "The dimension amount of train data (%d, %d) or the test data (%d, %d) does not match the expected (3, 2)\n",
            train_images->dimensions_size,
            train_labels->dimensions_size,
            test_images->dimensions_size,
            test_labels->dimensions_size
        );
        return 0;
    }
    return 1;
}

float random_float_unit(void) {
    return ((float)rand()/(float)(RAND_MAX-1));
}

struct fann_train_data * create_data_from_idx(struct idx_struct * images, struct idx_struct * labels, int digit) {
    struct fann_train_data * data;
    {
        unsigned int num_data = labels->dimensions[0];
        unsigned int num_input = images->dimensions[1] * images->dimensions[2];
        data = fann_create_train(num_data, num_input, 1);
        if (!data) {
            printf("Could not allocate training data\n");
            return NULL;
        } else if (labels->dimensions_size != 1) {
            printf("Expected labels to have 1 dimension, got %d\n", labels->dimensions_size);
            return NULL;
        } else if (labels->dimensions[0] != images->dimensions[0]) {
            printf("Expected labels dimension (%d) to be the same as the image dimension (%d)\n", labels->dimensions[0], images->dimensions[0]);
            return NULL;
        }
        unsigned int data_index = 0;
        for (int i = 0; i < num_data; i++) {
            float value;
            for (int j = 0; j < num_input; j++) {
                value = images->data[i * num_input + j] / 255.0;
                data->input[i][j] = value;
            }
            value = (((int) labels->data[i]) == ((int) digit)) ? 1 : 0;

            for (int j = 0; j < 1; j++) {
                data->output[i][j] = value;
            }
        }
    }
    return data;
}

float evaluate_network(struct fann * ann, struct fann_train_data * data) {
    unsigned int correct_guess_count = 0;
    unsigned int incorrect_guess_count = 0;
    for (int i = 0; i < data->num_data; i++) {
        fann_type * result_ptr = fann_run(ann, data->input[i]);
        fann_type expected = data->output[i][0];
        fann_type result = result_ptr[0];

        int is_true_positive = result > 0.5 && expected > 0.5;
        int is_true_negative = result <= 0.5 && expected <= 0.5;
        // int is_false_positive = result > 0.5 && expected <= 0.5;
        // int is_false_negative = result <= 0.5 && expected > 0.5;

        if (is_true_positive || is_true_negative) {
            correct_guess_count++;
        } else {
            incorrect_guess_count++;
        }
    }
    return (float) (correct_guess_count) / (float) ((float) correct_guess_count + (float) incorrect_guess_count);
}

struct fann_train_data * create_data_subset(struct fann_train_data * data, unsigned int subset_size, int equalize) {
    struct fann_train_data * result = fann_create_train(data->num_data > subset_size ? subset_size : data->num_data, data->num_input, data->num_output);

    int continue_count = 0;
    for (int i = 0; i < subset_size; i++) {
        unsigned int index;
        if (subset_size >= data->num_data) {
            index = i;
        } else {
            index = random_float_unit() * data->num_data;
        }

        if (index < 0 || index >= data->num_data) {
            continue_count++;
            if (continue_count > data->num_data * 2) {
                printf("Failed at finding index multiple times\n");
                fann_destroy_train(result);
                return NULL;
            }
            i--;
            continue;
        }

        if (data->num_output == 1 && data->output[index][0] == 0 && (random_float_unit() < 0.92)) {
            if (equalize) {
                continue_count++;
                if (continue_count > data->num_data * 2) {
                    printf("Failed at equalizing data multiple times\n");
                    fann_destroy_train(result);
                    return NULL;
                }
                i--;
                continue;
            }
        }

        for (int j = 0; j < data->num_input; j++) {
            result->input[i][j] = data->input[index][j];
        }
        for (int j = 0; j < data->num_output; j++) {
            result->output[i][j] = data->output[index][j];
        }
    }

    return result;
}

int main(int argn, char ** argv) {
    srand((unsigned int) time(0));

    int variant = argn >= 2 ? atoi(argv[1]) : 0;
    if (variant < 0 || variant > 7) {
        printf("Variant %d out of range\n", variant);
        return 1;
    }

    printf("Variant: %d\n", variant);

    struct fann_train_data * train_data[10];
    struct fann_train_data * test_data[10];
    struct idx_struct * train_images = create_idx_data_by_loading_file(source_type_train, input_type_image);
    struct idx_struct * train_labels = create_idx_data_by_loading_file(source_type_train, input_type_label);
    struct idx_struct * test_images = create_idx_data_by_loading_file(source_type_test, input_type_image);
    struct idx_struct * test_labels = create_idx_data_by_loading_file(source_type_test, input_type_label);
    int image_width;
    int image_height;
    {
        printf("Reading input idx files.\n");

        if (!validate_data_from_files(train_images, train_labels, test_images, test_labels)) {
            return 1;
        }

        printf("Creating dataset from file data.\n");
        for (int i = 0; i < 10; i++) {
            train_data[i] = create_data_from_idx(train_images, train_labels, i);
            test_data[i] = create_data_from_idx(test_images, test_labels, i);
            if (!train_data[i] || !test_data[i]) {
                return 1;
            }
        }
        image_width = train_images->dimensions[1];
        image_height = train_images->dimensions[2];
    }
    destroy_idx_data(train_images);
    destroy_idx_data(train_labels);

    {
        /*
        printf("Samples:\n");
        for (int i = 0; i < 3; i++) {
            int digit_index = (int)(10.0 * random_float_unit());
            int data_index = (int)((float) train_data[digit_index]->num_data * random_float_unit());

            printf("Sample %d from training images (is %d? %s):\n", data_index, digit_index, (train_data[digit_index]->output[data_index][0]) > 0.5 ? "Yes" : "No");
            print_grayscale_image(train_data[digit_index]->input[data_index], image_width, image_height);
        }
        for (int i = 0; i < 3; i++) {
            int digit_index = (int)(10.0 * random_float_unit());
            int data_index = (int)((float) test_data[digit_index]->num_data * random_float_unit());

            printf("Sample %d from training images (is %d? %s):\n", data_index, digit_index, test_data[digit_index]->output[data_index][0] > 0.5 ? "Yes" : "No");
            print_grayscale_image(test_data[digit_index]->input[data_index], image_width, image_height);
        }
        */
    }

    struct fann * ann[10];
    {
        printf(LOAD_NETWORKS ? "Loading networks.\n" : "Creating networks.\n");
        int layers[] = { image_width * image_height, 114, 1 };
        for (int i = 0; i < 10; i++) {
            if (LOAD_NETWORKS) {
                char buffer[256];
                snprintf(buffer, sizeof(buffer) - 1, "./output/%d-detecting-network-v%d.txt", i, variant);
                ann[i] = fann_create_from_file(buffer);
                if (!ann[i]) {
                    printf("Error: could not load network from \"%s\"\n", buffer);
                    return 1;
                }
            } else {
                enum fann_train_enum training_options[] = {FANN_TRAIN_INCREMENTAL, FANN_TRAIN_BATCH, FANN_TRAIN_INCREMENTAL, FANN_TRAIN_QUICKPROP, FANN_TRAIN_BATCH, FANN_TRAIN_BATCH, FANN_TRAIN_BATCH, FANN_TRAIN_QUICKPROP};
                enum fann_activationfunc_enum hidden_activation_options[] = {FANN_SIGMOID, FANN_LEAKY_RELU, FANN_LEAKY_RELU, FANN_COS, FANN_SIGMOID, FANN_SIN, FANN_SIN, FANN_SIGMOID};
                enum fann_activationfunc_enum output_activation_options[] = {FANN_ELLIOT, FANN_LINEAR, FANN_ELLIOT, FANN_GAUSSIAN, FANN_LINEAR, FANN_SIGMOID, FANN_LEAKY_RELU, FANN_LINEAR_PIECE};
                int dataset_size_options[] = {800, 500, 200, 500, 500, 800, 500, 650};
                int epoch_count_options[] = {10, 70, 70, 190, 250, 250, 310, 190};
                enum fann_errorfunc_enum errorfunc_options[] = {FANN_ERRORFUNC_TANH, FANN_ERRORFUNC_LINEAR, FANN_ERRORFUNC_TANH, FANN_ERRORFUNC_TANH, FANN_ERRORFUNC_TANH, FANN_ERRORFUNC_LINEAR, FANN_ERRORFUNC_LINEAR, FANN_ERRORFUNC_LINEAR};
                int hidden_layer_sizes[] = {104, 122, 122, 122, 31, 67, 49, 104};
                float randomize_range_options[] = {0.25, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.5};
                float learning_rate_options[] = {0.59, 0.23, 0.5, 0.5, 0.14, 0.59, 0.59, 0.77};

                layers[1] = hidden_layer_sizes[variant];

                ann[i] = fann_create_standard_array(sizeof(layers) / sizeof(int), layers);
                if (!ann[i]) {
                    printf("Error: could not create network %d\n", i);
                    return 1;
                }
                fann_set_training_algorithm(ann[i], training_options[variant]);
                fann_set_activation_function_hidden(ann[i], hidden_activation_options[variant]);
                fann_set_activation_function_output(ann[i], output_activation_options[variant]);
                dataset_size = dataset_size_options[variant];
                epoch_count = epoch_count_options[variant];
                fann_set_train_error_function(ann[i], errorfunc_options[variant]);
                float randomize_range = randomize_range_options[variant];
                fann_set_learning_rate(ann[i], learning_rate_options[variant]);
                fann_randomize_weights(ann[i], -randomize_range, randomize_range);
            }
        }
    }

    if (TRAIN_NETWORKS) {
        printf("Evaluating networks.\n");
        for (int i = 0; i < 10; i++) {
            float performance = evaluate_network(ann[i], test_data[i]);
            printf("Network %d: Performance: %.3f %%\n", i, 100.0 * performance);
        }

        printf("Training networks.\n");
        {
            for (int i = 0; i < 10; i++) {
                for (int step_id = 0; step_id < 20; step_id++) {
                    struct fann_train_data * subdata = create_data_subset(train_data[i], dataset_size, 1);
                    float dataset_positivity;
                    {
                        int positive = 0;
                        int negative = 0;
                        for (int i = 0; i < subdata->num_data; i++) {
                            if (subdata->output[i][0] != 0) {
                                positive++;
                            } else {
                                negative++;
                            }
                        }
                        dataset_positivity = (float) positive / (float) (positive + negative);
                    }
                    printf("Network %d/%d - Step %d/%d - (Dataset positivity: %.2f)\n", i, 10, step_id, 40, dataset_positivity);

                    fann_train_on_data(
                        ann[i],
                        subdata,
                        epoch_count, // max epochs
                        epoch_count / 2, // epochs between reports
                        0.0001 // desired error
                    );

                    fann_destroy_train(subdata);
                }
            }
        }
    }

    printf("Evaluating networks.\n");
    for (int i = 0; i < 10; i++) {
        float performance = evaluate_network(ann[i], test_data[i]);
        printf("Network %d: Performance: %.3f %%\n", i, 100.0 * performance);
    }

    printf("Evaluating all models together.\n");
    {
        int correct_guesses = 0;
        int incorrect_guesses = 0;

        int input_size = image_width * image_height;
        int output_size = 10;
        fann_type * input = calloc(input_size, sizeof(fann_type));
        fann_type * output = calloc(output_size, sizeof(fann_type));
        int test_count = test_labels->dimensions[0];

        int has_shown = 0;

        for (int pair_id = 0; pair_id < test_count; pair_id++) {
            for (int i = 0; i < input_size; i++) {
                input[i] = (double) test_images->data[pair_id * input_size + i] / 255.0;
            }

            int highest_id = 0;
            for (int i = 0; i < 10; i++) {
                fann_type * single_output = fann_run(ann[i], input);
                output[i] = single_output[0];
                if (i == 0 || output[i] > output[highest_id]) {
                    highest_id = i;
                }
            }

            if (highest_id == test_labels->data[pair_id]) {
                correct_guesses++;
            } else {
                incorrect_guesses++;
            }

            /*
                if (has_shown < 3 && highest_id != test_labels->data[pair_id]) {
                    has_shown++;
                    print_grayscale_image(input, image_width, image_height);
                    printf("At train id %d - Expected: %d, got: %d (", pair_id, test_labels->data[pair_id], highest_id);
                    for (int i = 0; i < 10; i++) {
                        printf(" %.2f", output[i]);
                    }
                    printf(")\n");
                }
            */
        }

        printf("Guessed %d correctly and %d incorrectly out of %d. Performance: %.2f %%\n", correct_guesses, incorrect_guesses, correct_guesses + incorrect_guesses, 100.0 * (float) correct_guesses / (float)(correct_guesses + incorrect_guesses));

        free(input);
        free(output);
    }


    if (SAVE_NETWORKS) {
        printf("Saving networks.\n");
        for (int i = 0; i < 10; i++) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer) - 1, "./output/%d-detecting-network-v%d.txt", i, variant);
            fann_save(ann[i], buffer);
        }
    }

    destroy_idx_data(test_images);
    destroy_idx_data(test_labels);

    printf("Freeing memory\n");
    for (int i = 0; i < 10; i++) {
        fann_destroy_train(train_data[i]);
        fann_destroy_train(test_data[i]);
        fann_destroy(ann[i]);
    }

    return 0;
}