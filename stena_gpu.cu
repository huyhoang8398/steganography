#include "stena_gpu.cuh"
#include "stena_cpu.h"

__device__ __constant__ float cfilter[32 * 32]; // constant memory for filter. This gives 2x better speed
// pre-allocate data array on gpu, trade memory usage for speed 
namespace StenaGpu {
    // variables with the letter 'g' in front are arrays in GPU memory
    int sizeV = 0, sizeIndexes = 0, sizegV = 0, sizeSort = 0, sizeFilter = 0, sizePixels = 0;
    byte *gpixels_alloc = nullptr;
    float *gfilter_alloc = nullptr, *gV_alloc = nullptr;
    float *V_alloc = nullptr;
    pair_fi *sortData_alloc = nullptr;
    int *bestIndexes_alloc = nullptr;

    void initPixels(int n) {
        if (gpixels_alloc == nullptr) {
            cudaMalloc(&gpixels_alloc, 4000 * 2200 * 3 * sizeof(byte));
            sizeFilter = 4000 * 2200;
        }
        if (sizePixels < n) {
            cudaFree(gpixels_alloc);
            cudaMalloc(&gpixels_alloc, n * 3 * sizeof(byte));
            sizePixels = n;
        }
    }

    void initFilter(int n) {
        if (gfilter_alloc == nullptr) {
            cudaMalloc(&gfilter_alloc, 32 * 32 * sizeof(float));
            sizeFilter = 32 * 32;
        }
        if (sizeFilter < n) {
            cudaFree(gfilter_alloc);
            cudaMalloc(&gfilter_alloc, n * sizeof(float));
            sizeFilter = n;
        }
    }

    void initgV(int n) {
        if (gV_alloc == nullptr) {
            cudaMalloc(&gV_alloc, 4000 * 2200 * sizeof(float));
            sizegV = 4000 * 2200;
        }
        if (sizegV < n) {
            cudaFree(gV_alloc);
            cudaMalloc(&gV_alloc, n * sizeof(float));
            sizegV = n;
        }
    }

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
using namespace StenaGpu;

//*****************
void stenaInitCuda(const PPMImage *image, const Filter &filter, int n) {
    int imgH = image->height, imgW = image->width;
    int filtH = filter.height, filtW = filter.width;

    initPixels(imgH * imgW);
    initFilter(filtH * filtW);
    initgV(imgH * imgW);
    initV(filtH * filtW);
    initIndexes(n);
}

//---
// We use x for rows, y for columns, like in stena_cpu
// Filter width and height must be < 32 (since TILE_DIM is constant).
#define TILE_DIM (32)
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)

__global__
void myconv2dCuda(const int imgH, const int imgW, const byte *pixels,
                  const int filtH, const int filtW, const float *filter,
                  float *V) {
    // Each block has size 32x32 (threads). Each block will process a group of columns.
    //   Each thread correspond to one pixel, the filter's top-left corner is placed at (myrow, mycol).
    //   So, thread (tidx,tidy) has input (myrow, mycol) and output to pixel V[myrow + filtH/2][mycol + filtW/2].
    //     -> Each block process (TILE_DIM - filtW + 1) columns (draw an image to imagine).
    //     -> To process imgW column, need roundup(imgW / (TILE_DIM-filtW+1)) blocks.
    // To process entire image, each block loop over rows:
    //   process rows 0...x, x+1...2x, 2x+1...3x, ...; where x = TILE_DIM - filtH + 1

    __shared__ float smem[TILE_DIM][TILE_DIM];

    // rowsPerBlock = TILE_DIM - filtH + 1; Formula: roundup(a/b) = (a + b - 1) / b.
    const int loop = (imgH + (TILE_DIM - filtH + 1) - 1) / (TILE_DIM - filtH + 1),
            stride = TILE_DIM - filtH + 1,
            halfH = filtH / 2, halfW = filtW / 2;

    // colsPerBlock = TILE_DIM - filtW + 1
    const int mycol = blockIdx.x * (TILE_DIM - filtW + 1) + tidy,
            mycolOut = mycol + halfW;
    int myrow = tidx, myrowOut = tidx + halfH;

    for (int t = 0; t < loop; t++) {
        __syncthreads();

        // load image data to shared memory
        if (mycol >= imgW || myrow >= imgH) smem[tidx][tidy] = 0;
        else {
            int pixelPos = 3 * cell(myrow, mycol, imgW);
            smem[tidx][tidy] = pixels[pixelPos] + pixels[pixelPos + 1]; // red, green
        }
        __syncthreads();

        // convolute. Note that all threads in a warp access the same filter[cell(i,j,filtW)] at all steps.
        // So, we use constant memory for better speed.
        if (tidx < TILE_DIM - filtH + 1 && tidy < TILE_DIM - filtW + 1 &&
            // the top-left corner of the filter is put here, and it must fit inside the tile.
            myrowOut < imgH - halfH && mycolOut < imgW - halfW) // It must output to a pixel position inside the image
        {
            float tmp = 0;
            for (int i = 0; i < filtH; i++)
                for (int j = 0; j < filtW; j++)
                    tmp += smem[tidx + i][tidy + j] * filter[cell(i, j, filtW)];

            V[cell(myrowOut, mycolOut, imgW)] = tmp;
        }

        // update indexes
        myrow += stride;
        myrowOut += stride;
    }
}

void stenaConvCuda(const PPMImage *image, const Filter &filter, float **outputgV) {
    int imgH = image->height, imgW = image->width;
    int filtH = filter.height, filtW = filter.width;

    byte *gdata = gpixels_alloc;;
    cudaMemcpy(gdata, image->data, imgH * imgW * 3 * sizeof(byte), cudaMemcpyHostToDevice);

    // constant memory -> 2x faster than global memory, which is commented out
    //float* gfilter = gfilter_alloc;
    //cudaMemcpy(gfilter, filter.data, filtH * filtW * sizeof(float), cudaMemcpyHostToDevice);    
    float *gfilter;
    cudaGetSymbolAddress((void **) &gfilter, cfilter);
    cudaMemcpyToSymbol(cfilter, filter.data, filtH * filtW * sizeof(float));

    float *gV = gV_alloc;
    cudaMemset(gV, 0, imgH * imgW * sizeof(float)); // make sure all border pixels are 0
    int columnsPerBlock = TILE_DIM - filtW + 1;     // each block can process x columns -> need roundup(W/x) blocks to cover all columns
    dim3 grid((imgW + columnsPerBlock - 1) / columnsPerBlock, 1, 1);
    dim3 block(TILE_DIM, TILE_DIM, 1);
    myconv2dCuda << < grid, block, 2 * filtH * filtW * sizeof(float) >> >
    (imgH, imgW, gdata, filtH, filtW, gfilter, gV);
    cudaDeviceSynchronize();

    *outputgV = gV;
}

//*****************
void stenaCuda(const string s, const PPMImage *image, const Filter &filter, PPMImage **result, Bencher *bench) {
    const int imgH = image->height, imgW = image->width;
    const int filtH = filter.height, filtW = filter.width;
    const int n = s.length();
    MyTimer timer;

    if (filtH % 2 == 0 || filtW % 2 == 0) {
        printf("This project only supports odd-size filters. Even-size DLC 999$");
        exit(-1);
    }

    if (8 * n > imgH * imgW) {
        cout << "String is longer than this image. Not supported\n";
        exit(-1);
    }

    //****
    timer.startCounter();
    stenaInitCuda(image, filter, n); // allocate arrays on CPU and GPU
    if (bench != nullptr) bench->timeInit = timer.getCounter();

    //*****
    timer.startCounter();
    float *gV;
    stenaConvCuda(image, filter, &gV); // convolution and store result in gV, a GPU array
    if (bench != nullptr) bench->timeConv = timer.getCounter();

    //*****
    timer.startCounter();
    float *V = V_alloc;
    cudaMemcpy(V, gV, imgH * imgW * sizeof(float), cudaMemcpyDeviceToHost);

    int *bestIndexes = bestIndexes_alloc;
    pair_fi *sortData = sortData_alloc;
    for (int i = 0; i < imgH * imgW; i++) sortData[i] = pair_fi(V[i], i);
    findLargestValues(imgH, imgW, sortData, 8 * n, bestIndexes);
    if (bench != nullptr) bench->timeSort = timer.getCounter();

    //*****
    timer.startCounter();
    hideString(s, image, bestIndexes, result);
    if (bench != nullptr) bench->timeOutput = timer.getCounter();
}

string stenaInvCuda(PPMImage *image, const Filter &filter, const int n) {
    const int imgH = image->height, imgW = image->width;
    if (n > imgH * imgW) {
        cout << "String is longer than this image. Can't recover string\n";
        exit(-1);
    }

    //****    
    // everything is the same as stenaCuda, only the final step is different (getString() instead of hideString())
    stenaInitCuda(image, filter, n);

    float *gV;
    stenaConvCuda(image, filter, &gV); // convolution and store result in gV, a GPU array    

    //*****
    float *V = V_alloc;
    cudaMemcpy(V, gV, imgH * imgW * sizeof(float), cudaMemcpyDeviceToHost);

    //*****
    int *bestIndexes = bestIndexes_alloc;
    pair_fi *sortData = sortData_alloc;
    for (int i = 0; i < imgH * imgW; i++) sortData[i] = pair_fi(V[i], i);
    findLargestValues(imgH, imgW, sortData, 8 * n, bestIndexes);

    //*****
    return getString(n, image, bestIndexes);
}