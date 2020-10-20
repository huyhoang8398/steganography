#pragma once
#define RGB_COMPONENT_COLOR 255

struct PPMPixel {
     unsigned char red, green, blue;

     // add == and != operators to make coding easier
     bool operator == (const PPMPixel& other) const {
         return (red == other.red) && (green == other.green) && (blue == other.blue);         
     }

     bool operator != (const PPMPixel& other) const {
         return !(*this == other);
     }
};

struct PPMImage {
     int width = 0, height = 0;
     PPMPixel *data = nullptr;     
};

PPMImage *readPPM(const char *filename);
void writePPM(const char *filename, PPMImage *img);

void free(PPMImage* image);