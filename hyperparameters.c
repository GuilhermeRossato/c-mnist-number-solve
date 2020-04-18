
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
    float learning_rate;
};

struct hyperparameters * create_hyperparameters(int input_x, int input_y, int output_size) {
    struct hyperparameters * h = malloc(sizeof(struct hyperparameters));
    h->layers_size = 4;
    h->layers = malloc(sizeof(unsigned int) * (h->layers_size + 1));
    h->layers[0] = input_x * input_y;
    h->layers[1] = 49;
    h->layers[2] = 10;
    h->layers[3] = output_size;
    h->is_random_weight = 1;
    h->training_algorithm = TRAINING_ALGORITHM_INCREMENTAL;
    h->learning_rate = 0.9;
    return h;
}

void destroy_hyperparameters(struct hyperparameters * h) {
    free(h->layers);
    free(h);
}

struct fann * create_network_with_hyperparameters(struct hyperparameters * h) {
    struct fann * ann = fann_create_standard_array(h->layers_size, h->layers);

    fann_set_training_algorithm(ann, h->training_algorithm);
    fann_set_learning_rate(ann, h->learning_rate);

    if (h->is_random_weight) {
        fann_randomize_weights(ann, -0.1, 0.1);
    } else {
        // Uses Widrow + Nguyen's algorithm, but that requires data.
        // fann_init_weight(ann, data);
    }

    return ann;
}

void print_hyperparameters(struct hyperparameters * h) {
    printf("Hyperparameters:\n");
    printf("%d layers: ", h->layers_size);
    for (int i = 0; i < h->layers_size; i++) {
        printf("%d ", h->layers[i]);
    }
    printf("\nit %s randomly initialized\n", h->is_random_weight ? "is" : "is not");
    printf("Training algorithm: %d\n", h->training_algorithm);
    printf("Learning Rate: %f\n\n", h->learning_rate);
}