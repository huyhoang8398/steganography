#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include "ppm_lib.h"
#include "my_timer.h"

using std::string;
using std::cin;
using std::cout;
using std::pair;
using std::greater;
using std::vector;
using std::ifstream;
using std::ofstream;

typedef unsigned char byte;
typedef pair<int, int> pair_ii;
typedef pair<float, int> pair_fi;

// get position in 1D array of pixel at row i, column j, in image with width w
#define cell(i,j,w) ((i)*(w) + (j))

// struct to store benchmark results of a steganographie function
struct Bencher {
    double timeInit=0, timeConv=0, timeSort=0, timeOutput=0;
    Bencher() {}

    double totalTime() {
        return timeInit + timeConv + timeSort + timeOutput;
    }

    Bencher operator + (const Bencher& other) const {
        Bencher res;
        res.timeInit = timeInit + other.timeInit;
        res.timeConv = timeConv + other.timeConv;
        res.timeSort = timeSort + other.timeSort;
        res.timeOutput = timeOutput + other.timeOutput;
        return res;
    }
};

static void print(Bencher bench) {
    cout << "time init = " << bench.timeInit << '\n';
    cout << "time conv = " << bench.timeConv << '\n';
    cout << "time sort = " << bench.timeSort << '\n';
    cout << "time output = " << bench.timeOutput << '\n';
}