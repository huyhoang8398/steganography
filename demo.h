#pragma once

#include "definitions.h"
#include "stena_cpu.h"
#include "stena_gpu.cuh"

// This function is used to hide a message into a file.
// The user first input the message and the image to hide.
// Finally, the "filter seed" acts as a password for the image.
// It it used to generate a filter, and the same seed will correspond to the same filter.
// The function will write the new image (hidden with the message) into a new file.
void demoEncrypt() {
    string mess, inputFile, outputFile;
    int seed;
    Filter filter;

    cout << "Input information below\n";
    cout << "Message: ";
    getline(cin, mess);
    cout << "Input file name: ";
    getline(cin, inputFile);
    cout << "Filter seed ( > 0): ";
    cin >> seed;
    cin.ignore();

    if (seed <= 0) {
        cout << "Seed is <= 0. Encryption cancelled\n";
        return;
    }

    int index = inputFile.find(".ppm");
    outputFile = inputFile;
    outputFile.replace(index, 4, "_stena.ppm");

    PPMImage *image, *encrypted;
    image = readPPM(inputFile.c_str());
    filter = Filter(25, 25, seed);
    mess += char(0);
    stenaCuda(mess, image, filter, &encrypted);
    writePPM(outputFile.c_str(), encrypted);
    cout << "Encrypted in file " << outputFile << "\n";
}

// This function is used to get the hidden message from a file.
// The user need to input the image and the filter seed (password).
// Finally, it will output the hidden message to screen.
void demoDecrypt() {
    string mess, inputFile;
    int seed;
    Filter filter;

    cout << "Input information below\n";
    cout << "Input file name: ";
    getline(cin, inputFile);
    cout << "Filter seed (same as encryption): ";
    cin >> seed;
    cin.ignore();

    if (seed <= 0) {
        cout << "Seed is <= 0. Decryption cancelled\n";
        return;
    }

    PPMImage *image;
    image = readPPM(inputFile.c_str());
    filter = Filter(25, 25, seed);
    mess = stenaInvCuda(image, filter, (image->width) * (image->height) / 8);
    cout << mess << "\n";
}

