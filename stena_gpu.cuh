#pragma once
#include "definitions.h"
#include "filter.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void stenaCuda(const string s, const PPMImage* image, const Filter& filter, PPMImage** result, Bencher* bench=nullptr);
string stenaInvCuda(PPMImage* image, const Filter& filter, const int n);