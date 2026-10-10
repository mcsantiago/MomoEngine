// stb_image and stb_image_write are header-only: this translation unit is the
// single place their implementations get compiled.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO_WRITE
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
