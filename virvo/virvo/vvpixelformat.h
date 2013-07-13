// vvpixelformat.h


#ifndef VV_PIXEL_FORMAT_H
#define VV_PIXEL_FORMAT_H


#include "vvexport.h"


namespace virvo
{


    // Pixel formats for color buffers and images
    enum EColorFormat
    {
        CF_UNSPECIFIED,
        CF_R8,
        CF_RG8,
        CF_RGB8,
        CF_RGBA8,
        CF_R16F,
        CF_RG16F,
        CF_RGB16F,
        CF_RGBA16F,
        CF_R32F,
        CF_RG32F,
        CF_RGB32F,
        CF_RGBA32F,
        CF_R16I,
        CF_RG16I,
        CF_RGB16I,
        CF_RGBA16I,
        CF_R32I,
        CF_RG32I,
        CF_RGB32I,
        CF_RGBA32I,
        CF_R16UI,
        CF_RG16UI,
        CF_RGB16UI,
        CF_RGBA16UI,
        CF_R32UI,
        CF_RG32UI,
        CF_RGB32UI,
        CF_RGBA32UI,

        CF_BGR8,
        CF_BGRA8,
        CF_RGB10_A2,
        CF_R11F_G11F_B10F,

        CF_COUNT // Last!!!
    };


    // Pixel formats for depth/stencil buffers
    enum EDepthFormat
    {
        DF_UNSPECIFIED,

        DF_DEPTH16,
        DF_DEPTH24,
        DF_DEPTH32,
        DF_DEPTH32F,
        DF_DEPTH24_STENCIL8,
        DF_DEPTH32F_STENCIL8,

        DF_LUMINANCE8,      // not an OpenGL format!
        DF_LUMINANCE16,     // not an OpenGL format!
        DF_LUMINANCE32F,    // not an OpenGL format!

        DF_COUNT // Last!!!
    };


    struct PixelFormat
    {
        unsigned internalFormat;
        unsigned format;
        unsigned type;
        unsigned components;
        unsigned size; // per pixel in bytes
    };


    // Returns some information about the given color format
    VVAPI PixelFormat mapPixelFormat(EColorFormat format);


    // Returns some information about the given depth/stencil format
    VVAPI PixelFormat mapPixelFormat(EDepthFormat format);


    // Returns the size of a single pixel of the given format
    inline unsigned getPixelSize(EColorFormat format)
    {
        PixelFormat f = mapPixelFormat(format);
        return f.size;
    }


    // Returns the size of a single pixel of the given format
    inline unsigned getPixelSize(EDepthFormat format)
    {
        PixelFormat f = mapPixelFormat(format);
        return f.size;
    }


} // namespace virvo


#endif
