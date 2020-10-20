#pragma once
#include "definitions.h"
#include "filter.h"

void myconv2d(const int imgH, const int imgW, const PPMPixel* pixels,
    const int filtH, const int filtW, const float* filter,
    float* V);

void findLargestValues(int imgH, int imgW, pair_fi* sortData, int nbValues, int* bestIndexes);

void hideString(const string s, const PPMImage* image, int* bestIndexes, PPMImage** res);
void stena(const string s, const PPMImage* image, const Filter& filter, PPMImage** res, Bencher* bench=nullptr);

string getString(int len, const PPMImage* image, int* bestIndexes);
string stenaInv(PPMImage* image, const Filter& filter, const int n);
