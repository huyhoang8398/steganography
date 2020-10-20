#include "definitions.h"
#include "stena_cpu.h"
#include "stena_gpu.cuh"
#include "auto_tester.cuh"
#include "demo.h"

int main() {
    srand(time(NULL));
    
    int opt;
    while (true) {
        cout << "input one option\n";
        cout << "1: encrypt a message into a file\n";
        cout << "2: decrypt a message from a file\n";
        cout << "3: check correctness of GPU versions (automated testing)\n";
        cout << "4: check speed of both versions\n";
        cout << "5: create benchmark results\n";
        cout << "Other: quit\n";

        cin >> opt;
        cin.ignore(); // to flush the \n character caused by cin
        if (opt == 1) demoEncrypt();
        else if (opt == 2) demoDecrypt();
        else if (opt == 3) testCorrectness(100);
        else if (opt == 4) testSpeed(5);
        else if (opt == 5) {
            // create benchmark files
            for (int i = 1; i <= 15; i++)
            {
                benchmark(200, 1080, 1920, 2 * i + 1, 2 * i + 1, false);
                benchmark(200, 1080, 1920, 2 * i + 1, 2 * i + 1, true);
            }
        }
        else break;
        
        cout << "\n\n************\n";
    }    
    
    cout << "Program finished\n";
    return 0;
}


