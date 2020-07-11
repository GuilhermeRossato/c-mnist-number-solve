#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define FANN_NO_DLL
#define FANN_NO_SEED
#define DISABLE_PARALLEL_FANN

#include "fann/doublefann.h"

#include "hyperparameters.h"
#include "hyperparameters.c"

#include "idx_reader.h"
#include "idx_reader.c"

#define IS_INPUT_ZERO_TO_ONE 1
#define IS_OUTPUT_ZERO_TO_ONE 1

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
const char letters_by_occupancy[95] = {' ', '`', '.', '-', '\'', ':', '_', ',', '^', '"', '~', ';', '!', '\\', '>', '/', '=', '*', '<', '+', 'r', 'c', 'v', 'L', '?', ')', 'z', '{', '(', '|', 'T', '}', 'J', '7', 'x', 's', 'u', 'n', 'Y', 'i', 'C', 'y', 'l', 't', 'F', 'w', '1', 'o', '[', ']', 'f', '3', 'I', 'j', 'Z', 'a', 'e', '5', 'V', '2', 'h', 'k', 'S', 'U', 'q', '9', 'P', '6', '4', 'd', 'K', 'p', 'A', 'E', 'b', 'O', 'G', 'm', 'R', 'H', 'X', 'N', 'M', 'D', '8', 'W', '#', '0', 'B', '$', '%', 'Q', 'g', '&', '@'};

void print_grayscale_image(double * image, uint32_t width, uint32_t height) {
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

struct fann_train_data * create_data_from_idx(struct idx_struct * images, struct idx_struct * labels, int label) {
    if (labels->dimensions_size != 1) {
        printf("Expected labels to have 1 dimension, got %d\n", labels->dimensions_size);
        return NULL;
    } else if (labels->dimensions[0] != images->dimensions[0]) {
        printf("Expected labels dimension (%d) to be the same as the image dimension (%d)\n", labels->dimensions[0], images->dimensions[0]);
        return NULL;
    }

    unsigned int num_data = labels->dimensions[0];
    unsigned int num_input = images->dimensions[1] * images->dimensions[2];

    struct fann_train_data * data = fann_create_train(num_data, num_input, 1);
    unsigned int data_index = 0;
    for (int i = 0; i < labels->dimensions[0]; i++) {
        // Write input
        for (int j = 0; j < images->dimensions[1] * images->dimensions[2]; j++) {
            float value = images->data[i * num_input + j] / 255.0;
            data->input[i][j] = IS_INPUT_ZERO_TO_ONE ? value : ((2.0 * value) - 1.0);
        }
        // Write output
        if (((int) labels->data[i]) == ((int) label)) {
            data->output[data_index][0] = 1;
        } else {
            data->output[data_index][0] = IS_INPUT_ZERO_TO_ONE ? 0 : -1;
        }
    }

    return data;
}

float evaluate_network(struct fann * ann, struct fann_train_data * data, int digit) {
    unsigned int correct_guess_count = 0;
    unsigned int incorrect_guess_count = 0;
    for (int i = 0; i < data->num_data; i++) {
        fann_type * result_ptr = fann_run(ann, data->input[i]);
        fann_type expected = data->output[i][0];

        fann_type result = result_ptr[0];

        if ( i < 10 ){
            printf("%f %f\n", result, expected);
        }

        if (IS_OUTPUT_ZERO_TO_ONE) {
            if (expected > 0.5 && result > 0.5) {
                correct_guess_count++;
            } else {
                incorrect_guess_count++;
            }
        } else {
            if (expected > 0 && result > 0) {
                correct_guess_count++;
            } else {
                incorrect_guess_count++;
            }
        }
    }
    return (correct_guess_count) / (correct_guess_count + incorrect_guess_count);
}

int main() {
    srand((unsigned int) time(0));

    struct fann_train_data * train_data[10];
    struct fann_train_data * test_data[10];
    int image_width;
    int image_height;
    {
        printf("Reading input idx files.\n");
        struct idx_struct * train_images = create_idx_data_by_loading_file(source_type_train, input_type_image);
        struct idx_struct * train_labels = create_idx_data_by_loading_file(source_type_train, input_type_label);
        struct idx_struct * test_images = create_idx_data_by_loading_file(source_type_test, input_type_image);
        struct idx_struct * test_labels = create_idx_data_by_loading_file(source_type_test, input_type_label);

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

        destroy_idx_data(train_images);
        destroy_idx_data(train_labels);
        destroy_idx_data(test_images);
        destroy_idx_data(test_labels);
    }

    {
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
    }

    struct fann * ann[10];
    {
        struct hyperparameters * h = create_hyperparameters(image_width, image_height, 1);

        printf("Creating networks.\n");
        for (int i = 0; i < 10; i++) {
            ann[i] = create_network_with_hyperparameters(h);
            fann_set_activation_function_hidden(ann[i], FANN_ELLIOT);
            fann_set_activation_function_output(ann[i], FANN_SIGMOID_STEPWISE);
        }

        destroy_hyperparameters(h);
    }

    printf("Evaluating networks.\n");
    for (int i = 0; i < 10; i++) {
        float performance = evaluate_network(ann[i], test_data[i], i);
        printf("Network %d: Performance: %.3f %%\n", i, 100.0 * performance);
    }

    printf("Training networks.\n");
    for (int i = 0; i < 10; i++) {
        printf("Network %d\n", i);
        fann_train_on_data(
            ann[i],
            train_data[i],
            20, // max epochs
            5, // epochs between reports
            0.001 // desired error
        );
    }

    printf("Evaluating networks.\n");
    for (int i = 0; i < 10; i++) {
        float performance = evaluate_network(ann[i], test_data[i], i);
        printf("Network %d: Performance: %.3f %%\n", i, 100.0 * performance);
    }

    printf("Freeing memory\n");
    for (int i = 0; i < 10; i++) {
        fann_destroy_train(train_data[i]);
        fann_destroy_train(test_data[i]);
        fann_destroy(ann[i]);
    }

    return 0;
}