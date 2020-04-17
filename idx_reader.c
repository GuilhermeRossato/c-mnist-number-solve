#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct idx_struct {
    uint8_t magic_zeros[2];
    uint8_t type_code;
    uint8_t dimensions_size;
    uint32_t * dimensions;
    uint32_t data_size;
    uint8_t * data;
};

// Advances and reads a single signed 32-bit integer from the file descriptor regardless of architecture and writes to result pointer, returns 0 if it fails, 1 if succeds
int8_t freadInt32BE(int32_t * result, FILE * f) {
    uint8_t buffer[sizeof(int32_t)];
    if (!result || !f || sizeof(int32_t) != fread((void *) buffer, 1, sizeof(int32_t), f))
        return 0;
    int32_t r = 0;
    int8_t shift = -8;
    for (int8_t i = sizeof(int32_t) - 1; i >= 0; i--)
        r += buffer[i] << (shift += 8);
    *result = r;
    return 1;
}

struct idx_struct * create_idx_from_file(const char * filename) {
    printf("Reading \"%s\"\n", filename);
    FILE * fileptr = fopen(filename, "rb");
    if (fileptr == NULL) {
        printf("Error: Could not open file: \"%s\"", filename);
        return NULL;
    }
    struct idx_struct * idx = malloc(sizeof(struct idx_struct));

    uint32_t bytes_to_read;

    bytes_to_read = 4;

    if (bytes_to_read != fread((void *) idx, sizeof(uint8_t), bytes_to_read, fileptr)) {
        printf(
            "Error: File reading failed to retrieve the first 4 bytes\n"
        );
        fclose(fileptr);
        free(idx);
        return NULL;
    } else if (idx->magic_zeros[0] != 0 || idx->magic_zeros[1] != 0) {
        printf(
            "Error: The magic number at the start of the files indicates incorrect file format. Expected 0x00 0x00, got 0x%02x 0x%02x\n",
            idx->magic_zeros[0],
            idx->magic_zeros[1]
        );
        fclose(fileptr);
        free(idx);
        return NULL;
    } else if (idx->type_code != 8) {
        printf(
            "Error: the type of data indicated by the code %d is not implemented. Expected 8 (uint8_t)\n",
            idx->type_code
        );
        fclose(fileptr);
        free(idx);
        return NULL;
    } else if (idx->dimensions_size < 0 || idx->dimensions_size > 3) {
        printf(
            "Error: The dimension size of %d is not implemented\n",
            idx->dimensions_size
        );
        fclose(fileptr);
        free(idx);
        return NULL;
    }

    idx->data_size = 1;
    idx->dimensions = malloc(idx->dimensions_size * sizeof(uint32_t));
    for (int i = 0; i < idx->dimensions_size; i++) {
        if (!freadInt32BE(&idx->dimensions[i], fileptr)) {
            printf("Error: File reading failed to retrieve the %d dimensions of the structure\n", idx->dimensions_size);
            fclose(fileptr);
            free(idx->dimensions);
            free(idx);
            return NULL;
        } else if (idx->dimensions[i] <= 0 || idx->dimensions[i] > 100000) {
            printf("Error: The %d th out of %d dimensions is outside bounds, expected [1 ~ 100000], got %08x\n", i, idx->dimensions_size, idx->dimensions[i]);
            fclose(fileptr);
            free(idx->dimensions);
            free(idx);
            return NULL;
        }
        idx->data_size *= idx->dimensions[i];
        if (idx->data_size <= 0 || idx->data_size > 104857600) {
            printf("Error: Total memory usage from file exceeds 100MB and is considered too big. Got %.1f MB\n", 1.0f * idx->data_size / (1024*1024));
            fclose(fileptr);
            free(idx->dimensions);
            free(idx);
            return NULL;
        }
    }

    bytes_to_read = sizeof(uint8_t) * idx->data_size;
    idx->data = malloc(bytes_to_read);
    if (bytes_to_read != fread((void *) idx->data, sizeof(uint8_t), idx->data_size, fileptr)) {
        printf("Error: File reading failed to retrieve the %d bytes of data\n", idx->data_size);
        fclose(fileptr);
        free(idx->data);
        free(idx->dimensions);
        free(idx);
        return NULL;
    }

    if (EOF != fgetc(fileptr)) {
        printf("Error: File has left-over data after reading its content\n", idx->data_size);
        fclose(fileptr);
        free(idx->data);
        free(idx->dimensions);
        free(idx);
        return NULL;
    }

    fclose(fileptr);
    return idx;
}