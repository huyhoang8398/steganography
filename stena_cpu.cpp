#pragma once
#include "stena_cpu.h"
#include "filter.h"
#include "quickselect.h"

// we pre-allocate necessary data variables to save time.
// Basically, trade a little memory for more speed.
namespace StenaCpu 
{
    int sizeV = 0, sizeIndexes = 0;
    float* V_alloc = nullptr;
    pair_fi* sortData_alloc = nullptr;
    int* bestIndexes_alloc = nullptr;
    
    void initV(int n) {
        if (V_alloc == nullptr) {
            V_alloc = new float[4000 * 2200]; // more than 4K
            sortData_alloc = new pair_fi[4000 * 2200];
            sizeV = 4000 * 2200;
        }
        if (sizeV < n) {
            delete[] V_alloc;
            delete[] sortData_alloc;
            V_alloc = new float[n];
            sortData_alloc = new pair_fi[n];
            sizeV = n;
        }
    }

    void initIndexes(int n) {
        n *= 8;
        if (bestIndexes_alloc == nullptr) {
            bestIndexes_alloc = new int[4000 * 2200];
            sizeIndexes = 4000 * 2200;
        }
        if (sizeIndexes < n) {
            delete[] bestIndexes_alloc;
            bestIndexes_alloc = new int[n];
            sizeIndexes = n;
        }
    }
}
using namespace StenaCpu;

//***************
void stenaInit(const PPMImage* image, int n)
{
    initV(image->width * image->height);
    initIndexes(n);
}

// This is the convolution function.
// The implementation is very very basic, just follow the definitions.
void myconv2d(const int imgH, const int imgW, const PPMPixel* pixels, 
              const int filtH, const int filtW, const float* filter, 
              float* V)
{              
    int i, j, k, u, v;    
    const int halfSizeX = filtH / 2, halfSizeY = filtW / 2;

    for (i = 0; i < imgH; i++) {        
        for (j = 0; j < imgW; j++) {
            int pixelIndex = cell(i, j, imgW); // 1D position of pixels at (i,j), in image with width imgW

            // Pixels at the border of V has value 0.
            if (i<halfSizeX || i>imgH - 1 - halfSizeX || j<halfSizeY || j>imgW - 1 - halfSizeY) {
                V[pixelIndex] = 0;
                continue;
            }

            int filtuv = 0;
            float value = 0;
            // convolute the filter from top->bot, left->right
            for (u = -halfSizeX; u <= halfSizeX; u++)
                for (v = -halfSizeY; v <= halfSizeY; v++)
                {                    
                    int convIndex = cell(i + u, j + v, imgW); 
                    value += (pixels[convIndex].red + pixels[convIndex].green) * filter[filtuv];                    
                    filtuv++;
                }

            V[cell(i, j, imgW)] = value;
        }
    }    
}

// find nbValues elements with largest values, then output their indexes to bestIndexes
void findLargestValues(int imgH, int imgW, pair_fi* sortData, int nbValues, int* bestIndexes)
{
    sortLargestOnly(imgH * imgW, sortData, nbValues);
    for (int i = 0; i < nbValues; i++) bestIndexes[i] = sortData[i].second;    
}

//*****************
// PPMImage** result should be uninitialized. It will be initialized here
void hideString(const string s, const PPMImage* image, int* bestIndexes, PPMImage** result)
{
    int imgH = image->height, imgW = image->width;
    int n = s.length();

    // The output image
    PPMImage* res = new PPMImage;
    res->height = imgH;
    res->width = imgW;
    res->data = new PPMPixel[imgH * imgW];
    memcpy(res->data, image->data, imgH * imgW * sizeof(PPMPixel));

    int currentBit = -1;
    for (int i = 0; i < n; i++) {
        // loop over 8 bit
        for (int j = 0; j < 8; j++) {
            currentBit++;
            int pixelIndex = bestIndexes[currentBit];

            // hide j-th bit of the character in the LSB of this pixel
            res->data[pixelIndex].blue &= ~byte(1); // byte(1) = 00000001 -> ~byte(1) = 11111110 -> this turns off bit 0
            res->data[pixelIndex].blue |= (s[i] >> j) & 1; // set bit 0 to the j-th bit of character s[i]
        }
    }

    *result = res;
}

// PPMImage** result is the pointer to the pointer of the output image
// It should be uninitialized. It will be initialized by this function
void stena(const string s, const PPMImage* image, const Filter& filter, PPMImage** result, Bencher* bench)
{    
    const int imgH = image->height, imgW = image->width;
    const int filtH = filter.height, filtW = filter.width;
    const int n = s.length();

    if (filtH % 2 == 0 || filtW % 2 == 0) {
        printf("This project only supports odd-size filters. Even-size DLC 999$");
        exit(-1);
    }
    if (8 * n > imgH * imgW) {
        cout << "String is longer than this image. Not supported\n";
        exit(-1);
    }

    //*****
    MyTimer timer;
    
    timer.startCounter();
    stenaInit(image, n);    
    if (bench != nullptr) bench->timeInit = timer.getCounter();
    
    //*****
    timer.startCounter();
    float* V = V_alloc;
    myconv2d(imgH, imgW, image->data, filtH, filtW, filter.data, V);
    if (bench != nullptr) bench->timeConv = timer.getCounter();
    
    //*****
    timer.startCounter();
    int* bestIndexes = bestIndexes_alloc;    
    pair_fi* sortData = sortData_alloc;
    for (int i = 0; i < imgH * imgW; i++) sortData[i] = pair_fi(V[i], i);

    findLargestValues(imgH, imgW, sortData, 8 * n, bestIndexes);
    if (bench != nullptr) bench->timeSort = timer.getCounter();

    //*****
    timer.startCounter();
    hideString(s, image, bestIndexes, result);
    if (bench!=nullptr) bench->timeOutput = timer.getCounter();         
}

//*****************
string getString(int n, const PPMImage* image, int* bestIndexes)
{
    string res = "";
    int currentBit = -1;

    for (int i = 0; i < n; i++)
    {
        char c = 0;
        for (int j = 0; j < 8; j++) {
            currentBit++;
            int pixelIndex = bestIndexes[currentBit];

            // Reads bit 0 at this pixel position and update c
            //   (x & 1) get bit 0 of x 
            //   c |= (1 << j) turns on j-th bit of c; c |= (0 << j) do nothing
            c |= (image->data[pixelIndex].blue & 1) << j;
        }
        if (c == 0) break;
        res += c;
    }

    return res;
}

string stenaInv(PPMImage* image, const Filter& filter, const int n)
{
    const int imgH = image->height, imgW = image->width;
    const int filtH = filter.height, filtW = filter.width;
    if (n > imgH * imgW) {
        cout << "String is longer than this image. Can't recover string\n";
        exit(-1);
    }
    
    //**************
    stenaInit(image, n);

    //**************
    float* V = V_alloc;
    myconv2d(imgH, imgW, image->data, filtH, filtW, filter.data, V);
    
    //**************
    int* bestIndexes = bestIndexes_alloc;
    pair_fi* sortData = sortData_alloc;
    for (int i = 0; i < imgH * imgW; i++) sortData[i] = pair_fi(V[i], i);
    findLargestValues(imgH, imgW, sortData, 8 * n, bestIndexes);

    //**************
    return getString(n, image, bestIndexes);
}