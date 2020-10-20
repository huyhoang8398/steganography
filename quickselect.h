#pragma once
#include "definitions.h"

// https://en.wikipedia.org/wiki/Quickselect
// This implementation is based on Quicksort with a form of Hoare's partition
void quickselect(pair_fi* a, int l, int r, int K)
{
    // we only cares about the first K elements, so if a sorting range (l,r)
    // is outside that segments, we can ignore it
    if (l > K) return;
    int i = l, j = r;
    pair_fi pivot = a[l + rand() % (r - l + 1)]; // random pivot for best performance

    // sort increasingly:
    // all elements larger than pivot to the left
    // smaller than pivots to the right
    while (i <= j) {
        while (a[i] > pivot) i++; 
        while (a[j] < pivot) j--; 
               
        if (i <= j) {
            swap(a[i], a[j]);
            i++;
            j--;
        }
    }

    if (i < r) quickselect(a, i, r, K);
    if (l < j) quickselect(a, l, j, K);
}

// sort array so that K largest elements is at the beginning.
// remaining elements don't matter
void sortLargestOnly(int n, pair_fi* a, int K)
{
    quickselect(a, 0, n - 1, K);
}