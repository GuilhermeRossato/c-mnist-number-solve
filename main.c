#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define FANN_NO_DLL
#define FANN_NO_SEED
#define DISABLE_PARALLEL_FANN

#include "fann/doublefann.c"
#include "fann/include/fann.h"

#include "hyperparameters.h"
#include "hyperparameters.c"

#include "idx_reader.h"
#include "idx_reader.c"

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

void print_grayscale_image(double * image, uint32_t width, uint32_t height, double * expected, double * output) {
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
    if (expected) {
        printf("Real number: ");
        int maxValueId = 0;
        double maxValue = expected[maxValueId];
        for (int i = 1; i < 10; i++) {
            if (expected[i] > maxValue) {
                maxValueId = i;
                maxValue = expected[maxValueId];
            }
        }
        if (maxValue != 1.0f) {
            printf("%d (%.2f)\n", maxValueId, maxValue);
        } else {
            printf("%d\n", maxValueId);
        }
    }
    if (output) {
        printf("Predicted: ");
        int maxValueId = 0;
        double maxValue = output[maxValueId];
        for (int i = 1; i < 10; i++) {
            if (output[i] > maxValue) {
                maxValueId = i;
                maxValue = output[maxValueId];
            }
        }
        printf("%d (%.2f)\n", maxValueId, maxValue);
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

int main() {
    srand((unsigned int) time(0));

    printf("Reading input files.\n");

    struct idx_struct * train_images = create_idx_data_by_loading_file(source_type_train, input_type_image);
    struct idx_struct * train_labels = create_idx_data_by_loading_file(source_type_train, input_type_label);
    struct idx_struct * test_images = create_idx_data_by_loading_file(source_type_test, input_type_image);
    struct idx_struct * test_labels = create_idx_data_by_loading_file(source_type_test, input_type_label);

    if (!validate_data_from_files(train_images, train_labels, test_images, test_labels)) {
        return 1;
    }

    printf("Creating dataset from file data.\n");

    train_images->dimensions[0] /= 1;
    train_labels->dimensions[0] /= 1;
    test_images->dimensions[0] /= 1;
    test_labels->dimensions[0] /= 1;

    double ** train_input = malloc(sizeof(double *) * train_images->dimensions[0]);
    double ** train_output = malloc(sizeof(double *) * train_labels->dimensions[0]);
    double ** test_input = malloc(sizeof(double *) * test_images->dimensions[0]);
    double ** test_output = malloc(sizeof(double *) * test_labels->dimensions[0]);

    if (!train_input || !train_output || !test_input || !test_output) {
        printf("Could not allocate training data\n");
        return 1;
    }

    // Allocate memory for double inputs and outputs
    for (int i = 0; i < train_images->dimensions[0]; i++) {
        train_input[i] = malloc(sizeof(double) * (train_images->dimensions[1] * train_images->dimensions[2]));
        if (!train_input[i]) {
            printf("Could not allocate training inputs\n");
            return 1;
        }
        train_output[i] = malloc(sizeof(double) * 10);
        if (!train_output[i]) {
            printf("Could not allocate training outputs\n");
            return 1;
        }
    }
    for (int i = 0; i < test_images->dimensions[0]; i++) {
        test_input[i] = malloc(sizeof(double) * (test_images->dimensions[1] * test_images->dimensions[2]));
        if (!test_input[i]) {
            printf("Could not allocate test inputs\n");
            return 1;
        }
        test_output[i] = malloc(sizeof(double) * 10);
        if (!test_output[i]) {
            printf("Could not allocate test outputs\n");
            return 1;
        }
    }
    for (int i = 0; i < train_images->dimensions[0]; i++) {
        for (int j = 0; j < train_images->dimensions[1] * train_images->dimensions[2]; j++) {
            train_input[i][j] = 2.0 * train_images->data[i * (train_images->dimensions[1] * train_images->dimensions[2]) + j] / 255.0 - 1.0;
        }
        for (int j = 0; j < 10; j++) {
            train_output[i][j] = train_labels->data[i] == j ? 1.0 : -1.0;
        }
    }
    for (int i = 0; i < test_images->dimensions[0]; i++) {
        for (int j = 0; j < test_images->dimensions[1] * test_images->dimensions[2]; j++) {
            test_input[i][j] = 2.0 * test_images->data[i * (test_images->dimensions[1] * test_images->dimensions[2]) + j] / 255.0 - 1.0;
        }
        for (int j = 0; j < 10; j++) {
            test_output[i][j] = test_labels->data[i] == j ? 1.0 : -1.0;
        }
    }
    printf("Samples:\n");
    for (int i = 0; i < 3; i++) {
        printf("Sample %d from training images:\n", i);
        print_grayscale_image(train_input[i], train_images->dimensions[1], train_images->dimensions[2], train_output[i], 0);
    }
    for (int i = 0; i < 3; i++) {
        printf("Sample %d from test images:\n", i);
        print_grayscale_image(test_input[i], test_images->dimensions[1], test_images->dimensions[2], test_output[i], 0);
    }

    struct hyperparameters * h = create_hyperparameters(train_images->dimensions[1], train_images->dimensions[2], 10);

    printf("Creating network.\n");
    struct fann * ann = create_network_with_hyperparameters(h);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    struct fann_train_data * data = fann_create_train_pointer_array(
        train_images->dimensions[0], // Amount of training data
        train_images->dimensions[1] * train_images->dimensions[2], // Input size
        train_input,
        10, // Output size
        train_output
    );

    printf("Training network.\n");
    fann_train_on_data(
        ann,
        data,
        100, // max epochs
        5, // epochs between reports
        0.001 // desired error
    );

    int corrects = 0;
    int incorrects = 0;
    printf("Testing network with training data:\n");
    for (int i = 0; i < train_images->dimensions[0]; i++) {
        double * result = fann_run(ann, train_input[i]);
        double * expected = train_output[i];
        if (i < 5) {
            print_grayscale_image(train_input[i], train_images->dimensions[1], train_images->dimensions[2], expected, result);
        }

        int expectedMaxValueId = 0;
        double expectedMaxValue = result[expectedMaxValueId];
        int outputMaxValueId = 0;
        double outputMaxValue = expected[outputMaxValueId];
        for (int i = 1; i < 10; i++) {
            if (result[i] > expectedMaxValue) {
                outputMaxValueId = i;
                outputMaxValue = result[outputMaxValueId];
            }
            if (expected[i] > expectedMaxValue) {
                expectedMaxValueId = i;
                expectedMaxValue = expected[expectedMaxValueId];
            }
        }

        if (expectedMaxValueId == outputMaxValueId) {
            corrects++;
        } else {
            incorrects++;
        }
    }
    printf("Matches %d out of %d (%.2f %%)\n", corrects, corrects+incorrects, 100.0f * corrects / (corrects + incorrects));
    printf("Testing network with testing data:\n");
    corrects = 0;
    incorrects = 0;
    for (int i = 0; i < test_images->dimensions[0]; i++) {
        double * result = fann_run(ann, test_input[i]);
        double * expected = test_output[i];
        if (i < 5) {
            print_grayscale_image(test_input[i], test_images->dimensions[1], test_images->dimensions[2], expected, result);
        }

        int expectedMaxValueId = 0;
        double expectedMaxValue = result[expectedMaxValueId];
        int outputMaxValueId = 0;
        double outputMaxValue = expected[outputMaxValueId];
        for (int i = 1; i < 10; i++) {
            if (result[i] > expectedMaxValue) {
                outputMaxValueId = i;
                outputMaxValue = result[outputMaxValueId];
            }
            if (expected[i] > expectedMaxValue) {
                expectedMaxValueId = i;
                expectedMaxValue = expected[expectedMaxValueId];
            }
        }

        if (expectedMaxValueId == outputMaxValueId) {
            corrects++;
        } else {
            incorrects++;
        }
    }
    printf("Matches %d out of %d (%.2f %%)\n", corrects, corrects+incorrects, 100.0f * corrects / (corrects + incorrects));

    for (int i = 0; i < train_labels->dimensions[0]; i++) {
        free(train_input[i]);
        free(train_output[i]);
    }
    for (int i = 0; i < test_images->dimensions[0]; i++) {
        free(test_input[i]);
        free(test_output[i]);
    }

    fann_destroy_train(data);
    fann_destroy(ann);
    destroy_hyperparameters(h);
    free(train_input);
    free(train_output);
    free(test_input);
    free(test_output);
    destroy_idx_data(train_images);
    destroy_idx_data(train_labels);
    destroy_idx_data(test_images);
    destroy_idx_data(test_labels);

    return 0;
}