// vvpixelformat.cpp


#include "vvpixelformat.h"

#include <assert.h>

#include <GL/glew.h>


const virvo::PixelFormat kColorFormats[] =
{
    { 0, 0, 0, 0, 0 }, // unspecified

    { GL_R8,                    GL_RED,                 GL_UNSIGNED_BYTE,                   1,  1 },    // CF_R8
    { GL_RG8,                   GL_RG,                  GL_UNSIGNED_BYTE,                   2,  2 },    // CF_RG8
    { GL_RGB8,                  GL_RGB,                 GL_UNSIGNED_BYTE,                   3,  3 },    // CF_RGB8
    { GL_RGBA8,                 GL_RGBA,                GL_UNSIGNED_BYTE,                   4,  4 },    // CF_RGBA8
    { GL_R16F,                  GL_RED,                 GL_HALF_FLOAT,                      1,  2 },    // CF_R16F
    { GL_RG16F,                 GL_RG,                  GL_HALF_FLOAT,                      2,  4 },    // CF_RG16F
    { GL_RGB16F,                GL_RGB,                 GL_HALF_FLOAT,                      3,  6 },    // CF_RGB16F
    { GL_RGBA16F,               GL_RGBA,                GL_HALF_FLOAT,                      4,  8 },    // CF_RGBA16F
    { GL_R32F,                  GL_RED,                 GL_FLOAT,                           1,  4 },    // CF_R32F
    { GL_RG32F,                 GL_RG,                  GL_FLOAT,                           2,  8 },    // CF_RG32F
    { GL_RGB32F,                GL_RGB,                 GL_FLOAT,                           3, 12 },    // CF_RGB32F
    { GL_RGBA32F,               GL_RGBA,                GL_FLOAT,                           4, 16 },    // CF_RGBA32F
    { GL_R16I,                  GL_RED_INTEGER,         GL_INT,                             1,  2 },    // CF_R16I
    { GL_RG16I,                 GL_RG_INTEGER,          GL_INT,                             2,  4 },    // CF_RG16I
    { GL_RGB16I,                GL_RGB_INTEGER,         GL_INT,                             3,  6 },    // CF_RGB16I
    { GL_RGBA16I,               GL_RGBA_INTEGER,        GL_INT,                             4,  8 },    // CF_RGBA16I
    { GL_R32I,                  GL_RED_INTEGER,         GL_INT,                             1,  4 },    // CF_R32I
    { GL_RG32I,                 GL_RG_INTEGER,          GL_INT,                             2,  8 },    // CF_RG32I
    { GL_RGB32I,                GL_RGB_INTEGER,         GL_INT,                             3, 12 },    // CF_RGB32I
    { GL_RGBA32I,               GL_RGBA_INTEGER,        GL_INT,                             4, 16 },    // CF_RGBA32I
    { GL_R16UI,                 GL_RED_INTEGER,         GL_UNSIGNED_INT,                    1,  2 },    // CF_R16UI
    { GL_RG16UI,                GL_RG_INTEGER,          GL_UNSIGNED_INT,                    2,  4 },    // CF_RG16UI
    { GL_RGB16UI,               GL_RGB_INTEGER,         GL_UNSIGNED_INT,                    3,  6 },    // CF_RGB16UI
    { GL_RGBA16UI,              GL_RGBA_INTEGER,        GL_UNSIGNED_INT,                    4,  8 },    // CF_RGBA16UI
    { GL_R32UI,                 GL_RED_INTEGER,         GL_UNSIGNED_INT,                    1,  4 },    // CF_R32UI
    { GL_RG32UI,                GL_RG_INTEGER,          GL_UNSIGNED_INT,                    2,  8 },    // CF_RG32UI
    { GL_RGB32UI,               GL_RGB_INTEGER,         GL_UNSIGNED_INT,                    3, 12 },    // CF_RGB32UI
    { GL_RGBA32UI,              GL_RGBA_INTEGER,        GL_UNSIGNED_INT,                    4, 16 },    // CF_RGBA32UI

    { GL_RGB8,                  GL_BGR,                 GL_UNSIGNED_BYTE,                   3,  3 },    // CF_BGR8
    { GL_RGBA8,                 GL_BGRA,                GL_UNSIGNED_BYTE,                   4,  4 },    // CF_BGRA8
    { GL_RGB10_A2,              GL_RGBA,                GL_UNSIGNED_INT_10_10_10_2,         4,  4 },    // CF_RGB10_A2
    { GL_R11F_G11F_B10F,        GL_RGB,                 GL_UNSIGNED_INT_10F_11F_11F_REV,    3,  4 },    // CF_R11F_G11F_B10F
};


const virvo::PixelFormat kDepthFormats[] =
{
    { 0, 0, 0, 0, 0 }, // unspecified

    { GL_DEPTH_COMPONENT16,     GL_DEPTH_COMPONENT,     GL_UNSIGNED_INT,                    1,  2 },    // DF_DEPTH16
    { GL_DEPTH_COMPONENT24,     GL_DEPTH_COMPONENT,     GL_UNSIGNED_INT,                    1,  3 },    // DF_DEPTH24
    { GL_DEPTH_COMPONENT32,     GL_DEPTH_COMPONENT,     GL_UNSIGNED_INT,                    1,  4 },    // DF_DEPTH32
    { GL_DEPTH_COMPONENT32F,    GL_DEPTH_COMPONENT,     GL_FLOAT,                           1,  4 },    // DF_DEPTH32F
    { GL_DEPTH24_STENCIL8,      GL_DEPTH_STENCIL,       GL_UNSIGNED_INT_24_8,               2,  4 },    // DF_DEPTH24_STENCIL8
    { GL_DEPTH32F_STENCIL8,     GL_DEPTH_STENCIL,       GL_FLOAT_32_UNSIGNED_INT_24_8_REV,  2,  8 },    // DF_DEPTH32F_STENCIL8

    // aus vvibrclient.cpp...
    { GL_LUMINANCE8,            GL_LUMINANCE,           GL_UNSIGNED_BYTE,                   1,  1 },    // DF_LUMINANCE8
    { GL_LUMINANCE16,           GL_LUMINANCE,           GL_UNSIGNED_SHORT,                  1,  2 },    // DF_LUMINANCE16
    { GL_LUMINANCE32F_ARB,      GL_LUMINANCE,           GL_FLOAT,                           1,  4 },    // DF_LUMINANCE32F
};


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
virvo::PixelFormat virvo::mapPixelFormat(EColorFormat format)
{
    unsigned index = static_cast<unsigned>(format);

    assert( 0 <= index && index < CF_COUNT );

    return kColorFormats[index];
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
virvo::PixelFormat virvo::mapPixelFormat(EDepthFormat format)
{
    unsigned index = static_cast<unsigned>(format);

    assert( 0 <= index && index < DF_COUNT );

    return kDepthFormats[index];
}
