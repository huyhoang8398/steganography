#pragma once
#include "definitions.h"
#include <time.h>

// This class is used to represent a filter in this project.
// It contains the height, width, and the elements of the filter.
class Filter {
public:
    int width, height;
    float* data;

public:
    // This class contains dynamic data, so we use canonical form
    // (default constructor, copy constructor, destructor, copy assignment operator)
    // to ensure good memory management.
    Filter() { width = 0; height = 0; data = nullptr; }

    Filter(int w, int h, float* values) {
        width = w;
        height = h;
        data = new float[w * h];
        for (int i = 0; i < w * h; i++) data[i] = values[i];
    }

    // Create a filter using the provided seed. Same sed will create the same filter.
    // note: if randomSeed == -1, then a random seed will be generated.
    Filter(int w, int h, int randomSeed) {        
        width = w;
        height = h;
        data = new float[w * h];

        std::default_random_engine generator;        
        std::uniform_int_distribution<int> distribution(-10, 10);
        if (randomSeed == -1) generator.seed(time(NULL));
        else generator.seed(randomSeed);

        for (int i = 0; i < width * height; i++) data[i] = distribution(generator);
    }

    Filter(const Filter& other) {
        width = other.width;
        height = other.height;
        data = new float[width * height];
        for (int i = 0; i < width * height; i++) data[i] = other.data[i];
    }

    ~Filter() {
        delete[] data;
    }

    Filter operator = (const Filter& other) {
        if (this != &other) {
            delete[] data;
            width = other.width;
            height = other.height;
            data = new float[width * height];
            for (int i = 0; i < width * height; i++) data[i] = other.data[i];
        }
        return *this;
    }
};
