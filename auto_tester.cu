#include "auto_tester.cuh"

namespace AutoTester {
    string randomString(int n) {
        string res = "";
        for (int i = 1; i <= n; i++) res += char(65 + rand() % 26); // random uppercase string
        return res;
    }

    void makeRandomImage(PPMImage **imagePtr, int height = -1, int width = -1) {
        if (height <= 0) height = rand() % 2160 + 8;
        if (width <= 0) width = rand() % 3840 + 8;

        PPMImage *image = new PPMImage();
        image->height = height;
        image->width = width;
        image->data = new PPMPixel[height * width];
        for (int i = 0; i < height * width; i++) {
            image->data[i].red = rand() % 256;
            image->data[i].green = rand() % 256;
            image->data[i].blue = rand() % 256;
        }
        *imagePtr = image;
    }

    Filter makeRandomFilter(int height = -1, int width = -1) {
        if (height <= 0) height = rand() % 31;
        if (height % 2 == 0) height++;
        if (width <= 0) width = rand() % 31;
        if (width % 2 == 0) width++;

        return Filter(width, height, rand() % 1000);
    }
}
using namespace AutoTester;


void testCorrectness(int ntest, int imgH, int imgW, int filtH, int filtW) {
    PPMImage *input, *cpuEncrypt = nullptr, *gpuEncrypt = nullptr;
    Filter filter;
    string message, decodeCpu, decodeGpu;

    if (filtH >= 32 || filtW >= 32) {
        cout << "Only support filter size upto 31x31\n";
        return;
    }

    cout << "Testing in progress\n";

    for (int t = 1; t <= ntest; t++) {
        makeRandomImage(&input, imgH, imgW);
        imgH = input->height;
        imgW = input->width;

        filter = makeRandomFilter(filtH, filtW);
        message = randomString(rand() % (imgH * imgW / 8) + 1);

        //*** Checking that both encryption gives the same output
        stena(message, input, filter, &cpuEncrypt);
        stenaCuda(message, input, filter, &gpuEncrypt);

        for (int i = 0; i < imgH * imgW; i++)
            if (cpuEncrypt->data[i] != gpuEncrypt->data[i]) {
                cout << "GPU encrypt wrong result\n";
                exit(-1);
            }

        //*** Checking that both decryption gives the same output
        decodeCpu = stenaInv(cpuEncrypt, filter, message.length());
        if (decodeCpu != message) {
            cout << "CPU decrypt wrong\n";
            exit(-1);
        }
        decodeGpu = stenaInvCuda(gpuEncrypt, filter, message.length());
        if (decodeGpu != message) {
            cout << "GPU decrypt wrong\n";
            exit(-1);
        }

        free(input);
        free(cpuEncrypt);
        free(gpuEncrypt);
        if (t % 10 == 0) cout << t << " tests passed\n";
    }

    cout << "Both implementations are correct\n";
}

void testSpeed(int ntest, int imgH, int imgW, int filtH, int filtW) {
    PPMImage *input, *cpuEncrypt = nullptr, *gpuEncrypt = nullptr;
    Filter filter;
    string message, decodeCpu, decodeGpu;
    int curImgH, curImgW, curFiltH, curFiltW;
    Bencher benchCpu, benchGpu;

    if (filtH >= 32 || filtW >= 32) {
        cout << "Only support filter size upto 31x31\n";
        exit(-1);
    }

    for (int t = 1; t <= ntest; t++) {
        makeRandomImage(&input, imgH, imgW);
        curImgH = input->height;
        curImgW = input->width;

        filter = makeRandomFilter(filtH, filtW);
        curFiltH = filter.height;
        curFiltW = filter.width;
        message = randomString(rand() % (curImgH * curImgW / 8) + 1);

        //****
        cout << "Image height, width = " << curImgH << ", " << curImgW << "\n";
        cout << "Filter height, width = " << curFiltH << ", " << curFiltW << "\n";

        stena(message, input, filter, &cpuEncrypt, &benchCpu);
        cout << "Cpu run time: \n";
        print(benchCpu);

        stenaCuda(message, input, filter, &gpuEncrypt, &benchGpu);
        cout << "Gpu run time: \n";
        print(benchGpu);

        cout << "\n\n****************\n";

        free(input);
        free(cpuEncrypt);
        free(gpuEncrypt);
        if (t % 10 == 0) cout << t << " executions finished\n";
    }
}

void benchmark(int ntest, int imgH, int imgW, int filtH, int filtW, bool useCuda) {
    PPMImage *input, *cpuEncrypt = nullptr, *gpuEncrypt = nullptr;

    Filter filter;
    string message, decodeCpu, decodeGpu;
    Bencher bench, totalBench;

    if (filtH >= 32 || filtW >= 32) {
        cout << "Only support filter size upto 31x31\n";
        exit(-1);
    }

    for (int t = 1; t <= ntest; t++) {
        makeRandomImage(&input, imgH, imgW);

        filter = makeRandomFilter(filtH, filtW);
        message = randomString(rand() % (imgH * imgW / 8) + 1);

        //****

        if (!useCuda) stena(message, input, filter, &cpuEncrypt, &bench);
        else stenaCuda(message, input, filter, &gpuEncrypt, &bench);
        totalBench = totalBench + bench; // use this to store the total running time, then calculate the average time

        free(input);
        free(cpuEncrypt);
        free(gpuEncrypt);
        if (t % 10 == 0) cout << t << " executions finished\n";
    }

    totalBench.timeInit /= ntest;
    totalBench.timeConv /= ntest;
    totalBench.timeSort /= ntest;
    totalBench.timeOutput /= ntest;

    //*****
    string filename = "bench";
    if (!useCuda) filename += "_cpu"; else filename += "_gpu";
    filename += "_" + std::to_string(imgH) + "x" + std::to_string(imgW);
    filename += ".txt";

    ofstream fo(filename, std::ios_base::app);
    fo << "resolution: " << imgH << " " << imgW << "\n";
    fo << "filter: " << filtH << " " << filtW << "\n";
    fo << "time-init: " << totalBench.timeInit << "\n";
    fo << "time-conv: " << totalBench.timeConv << "\n";
    fo << "time-sort: " << totalBench.timeSort << "\n";
    fo << "time-output: " << totalBench.timeOutput << "\n";
    fo << "time-total: " << totalBench.totalTime() << "\n";
    fo << "\n\n************\n";

    fo.close();
}