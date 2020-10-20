#pragma once
#include "definitions.h"
#include "stena_cpu.h"
#include "stena_gpu.cuh"

// - This function test the correctness of our Stena implementation
// by generating images and filters with given sizes, and compare
// the results of the GPU version with CPU version as well as 
// checking that the encryption/decryption process outputs the original message.
// - If a parameter (imgH, imgW, ...) is <= 0, it's randomly generated.
// - Parameters: number of test, image height/width, filter height/width (must be odd number <= 31)
void testCorrectness(int ntest, int imgH = -1, int imgW = -1, int filtH = -1, int filtW = -1);

// - Parameters: number of test, image height/width, filter height/width (must be odd number <= 31).
// Number <= 0 means randomly generated
// Read function testCorrectness()'s description for more details
void testSpeed(int ntest, int imgH = -1, int imgW = -1, int filtH = -1, int filtW = -1);

// - Parameters: number of test, image height/width, filter height/width (must be odd number <= 31),
// use CUDA or not.
// This will benchmark ntest times (might takes hours) and store the average run time in a file.
void benchmark(int ntest, int imgH, int imgW, int filtH, int filtW, bool useCuda = false);