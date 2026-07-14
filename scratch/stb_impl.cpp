// The single translation unit that compiles stb_image's implementation.
// stb_image is a single-header, public-domain library (no linking, no build
// dependency) used ONLY for decoding JPG/PNG/BMP. The learning core
// (autograd / conv / BatchNorm / CIoU / training) stays pure standard library.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
