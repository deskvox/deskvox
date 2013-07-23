// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


#include "vvcompress.h"
#include "vvcompressedvector.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvimage.h"
#include "../vvpixelformat.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

#ifdef HAVE_SNAPPY
#include <snappy.h>
#endif

#ifdef HAVE_JPEGTURBO
#include <jpeglib.h>
#if !defined(LIBJPEG_TURBO_VERSION)
#undef HAVE_JPEGTURBO
#endif
#if !defined(JCS_EXTENSIONS) || !defined(JCS_ALPHA_EXTENSIONS)
#undef HAVE_JPEGTURBO
#endif
#endif


//--------------------------------------------------------------------------------------------------
// Snappy
//--------------------------------------------------------------------------------------------------


#ifdef HAVE_SNAPPY


bool virvo::encodeSnappy(std::vector<unsigned char>& data)
{
    if (data.empty())
        return true;

    std::vector<unsigned char> compressed(snappy::MaxCompressedLength(data.size()));

    size_t len = 0;

    snappy::RawCompress((char const*)&data[0], data.size(), (char*)&compressed[0], &len);

    compressed.resize(len);

    data.swap(compressed);

    return true;
}


bool virvo::decodeSnappy(std::vector<unsigned char>& data)
{
    if (data.empty())
        return true;

    size_t len = 0;

    if (!snappy::GetUncompressedLength((char const*)&data[0], data.size(), &len))
        return false;

    std::vector<unsigned char> uncompressed(len);

    if (!snappy::RawUncompress((char const*)&data[0], data.size(), (char*)&uncompressed[0]))
        return false;

    data.swap(uncompressed);

    return true;
}


#else


bool virvo::encodeSnappy(std::vector<unsigned char>& /*data*/)
{
    return false;
}


bool virvo::decodeSnappy(std::vector<unsigned char>& /*data*/)
{
    return false;
}


#endif


bool virvo::encodeSnappy(CompressedVector& data)
{
    if (data.getCompressionType() != Compress_None)
        return false;

    if (encodeSnappy(data.vector()))
    {
        data.setCompressionType(Compress_Snappy);
        return true;
    }

    return false;
}


bool virvo::decodeSnappy(CompressedVector& data)
{
    if (data.getCompressionType() != Compress_Snappy)
        return false;

    if (decodeSnappy(data.vector()))
    {
        data.setCompressionType(Compress_None);
        return true;
    }

    return false;
}



//--------------------------------------------------------------------------------------------------
// JPEG
//--------------------------------------------------------------------------------------------------


#ifdef HAVE_JPEGTURBO


#ifdef _MSC_VER
#pragma warning(disable: 4324) // structure was padded due to __declspec(align())
#pragma warning(disable: 4611) // interaction between '_setjmp' and C++ object destruction is non-portable
#endif


namespace
{


// Error handler object
struct JPEGErrorManager
{
    // Error handler object
    jpeg_error_mgr pub;
    // setjmp buffer
    jmp_buf jmpbuf;
};


static void JPEGErrorExit(j_common_ptr info)
{
    JPEGErrorManager* err = (JPEGErrorManager*)info->err;

    //fprintf(stderr, "JPEG error %d: \"%s\"\n",
    //    err->pub.msg_code, err->pub.jpeg_message_table[err->pub.msg_code]);

    longjmp(err->jmpbuf, 1);
}


static J_COLOR_SPACE JPEGGetColorSpace(virvo::PixelFormat format)
{
    switch (format)
    {
    case virvo::PF_R8:
    case virvo::PF_LUMINANCE8:
        return JCS_GRAYSCALE;
    case virvo::PF_RGB8:
        return JCS_EXT_RGB;
    case virvo::PF_RGBA8:
        return JCS_EXT_RGBA;
    case virvo::PF_BGR8:
        return JCS_EXT_BGR;
    case virvo::PF_BGRA8:
        return JCS_EXT_BGRA;
    default:
        return JCS_UNKNOWN;
    }
}


struct JPEGDecodeCleanup
{
    jpeg_decompress_struct* info;

    JPEGDecodeCleanup() : info(0)
    {
    }

    ~JPEGDecodeCleanup()
    {
        if (info)
        {
            jpeg_finish_decompress(info); // Finish decompression
            jpeg_destroy_decompress(info); // Destroy the JPEG decompression object
        }
    }
};


struct JPEGEncodeCleanup
{
    jpeg_compress_struct* info;

    JPEGEncodeCleanup() : info(0)
    {
    }

    ~JPEGEncodeCleanup()
    {
        if (info)
        {
            jpeg_finish_compress(info);
            jpeg_destroy_compress(info);
        }
    }
};


// Encode a raw image into a std::vector
struct JPEGDestinationManager : jpeg_destination_mgr
{
    std::vector<unsigned char>& buffer;

    JPEGDestinationManager(std::vector<unsigned char>& buffer) : buffer(buffer)
    {
        this->init_destination = &InitDestination;
        this->empty_output_buffer = &EmptyOutputBuffer;
        this->term_destination = &TermDestination;
    }

    static void InitDestination(j_compress_ptr cinfo)
    {
        JPEGDestinationManager* dest = (JPEGDestinationManager*)cinfo->dest;

        dest->buffer.resize(16 * 1024);

        dest->next_output_byte = &dest->buffer[0];
        dest->free_in_buffer = dest->buffer.size();
    }

    static boolean EmptyOutputBuffer(j_compress_ptr cinfo)
    {
        JPEGDestinationManager* dest = (JPEGDestinationManager*)cinfo->dest;

        size_t size = dest->buffer.size();

        dest->buffer.resize(size * 2);

        dest->next_output_byte = &dest->buffer[size];
        dest->free_in_buffer = dest->buffer.size() - size;

        return TRUE;
    }

    static void TermDestination(j_compress_ptr cinfo)
    {
        JPEGDestinationManager* dest = (JPEGDestinationManager*)cinfo->dest;

        dest->buffer.resize(dest->buffer.size() - dest->free_in_buffer);
    }
};


} // namespace


bool virvo::encodeJPEG(std::vector<unsigned char>& data, JPEGOptions const& options)
{
    J_COLOR_SPACE colorSpace = JPEGGetColorSpace(options.format);

    if (colorSpace == JCS_UNKNOWN)
        return false;

    // Get the pixel format of the uncompressed image
    virvo::PixelFormatInfo imageFormat = mapPixelFormat(options.format);

    // Destination buffer
    std::vector<unsigned char> compressed;

    // Master record for a compression instance
    jpeg_compress_struct info;
    // Error handler object
    JPEGErrorManager err;
    // Data destination object for compression
    JPEGDestinationManager dest(compressed);
    // RAII Clean up
    JPEGEncodeCleanup cleanup;

    //
    // Initialize error-management
    //

    info.err = jpeg_std_error(&err.pub);

    err.pub.error_exit = &JPEGErrorExit;

    //
    // NOTE:
    //
    // All objects need to be fully constructed before this line, so these objects
    // get destructed properly when an error occurs.
    //
    if (setjmp(err.jmpbuf))
    {
        return false;
    }

    //
    // Initialize a JPEG decompression object.
    //

    jpeg_create_compress(&info);

    cleanup.info = &info;

    //
    // Specify the destination of the compressed data
    //

    info.dest = &dest;

    //
    // Set image parameters
    //

    info.image_width = options.w;
    info.image_height = options.h;

    //
    // Set parameters for compression
    //

    info.input_components = imageFormat.components;
    info.in_color_space = colorSpace;

    jpeg_set_defaults(&info);
    jpeg_set_quality(&info, options.quality, TRUE);

    //
    // Start decompression
    //

    jpeg_start_compress(&info, TRUE);

    //
    // Compress the image
    //

    while (info.next_scanline < info.image_height)
    {
        unsigned char* row = &data[0] + info.next_scanline * options.pitch;

        jpeg_write_scanlines(&info, &row, 1);
    }

    //
    // Success!
    //

    data.swap(compressed);

    return true;
}


bool virvo::decodeJPEG(std::vector<unsigned char>& data, JPEGOptions& options)
{
    J_COLOR_SPACE colorSpace = JPEGGetColorSpace(options.format);

    if (colorSpace == JCS_UNKNOWN)
        return false;

    // Get the pixel format of the uncompressed image
    virvo::PixelFormatInfo imageFormat = mapPixelFormat(options.format);

    // Destination buffer
    std::vector<unsigned char> uncompressed;

    // Master record for a decompression instance
    jpeg_decompress_struct info;
    // Error handler object
    JPEGErrorManager err;
    // RAII Clean up
    JPEGDecodeCleanup cleanup;

    //
    // Initialize error-management
    //

    info.err = jpeg_std_error(&err.pub);

    err.pub.error_exit = &JPEGErrorExit;

    //
    // NOTE:
    //
    // All objects need to be fully constructed before this line, so these objects
    // get destructed properly when an error occurs.
    //
    if (setjmp(err.jmpbuf))
    {
        return false;
    }

    //
    // Initialize a JPEG decompression object.
    //

    jpeg_create_decompress(&info);

    cleanup.info = &info;

    //
    // Specify the source of the compressed data
    //

    jpeg_mem_src(&info, &data[0], (unsigned long)data.size());

    //
    // Read image parameters
    //

    jpeg_read_header(&info, TRUE);

    options.w = info.image_width;
    options.h = info.image_height;
    options.pitch = options.w * imageFormat.size;

    //
    // Set parameters for decompression
    //

    info.out_color_components = imageFormat.components;
    info.out_color_space = colorSpace;

    // prefer speed...
    info.do_block_smoothing = 0;
    info.do_fancy_upsampling = 0;

    //
    // Start decompression
    //

    jpeg_start_decompress(&info);

    //
    // Decompress the image
    //

    uncompressed.resize(options.pitch * options.h);

    while (info.output_scanline < info.output_height)
    {
        unsigned char* row = &uncompressed[0] + info.output_scanline * options.pitch;

        jpeg_read_scanlines(&info, &row, 1);
    }

    //
    // Success!
    //

    data.swap(uncompressed);

    return true;
}


#ifdef _MSC_VER
#pragma warning(default: 4324)
#pragma warning(default: 4611)
#endif


#else


bool virvo::encodeJPEG(std::vector<unsigned char>& /*data*/, JPEGOptions const& /*options*/)
{
    return false;
}


bool virvo::decodeJPEG(std::vector<unsigned char>& /*data*/, JPEGOptions& /*options*/)
{
    return false;
}


#endif


bool virvo::encodeJPEG(CompressedVector& data, JPEGOptions const& options)
{
    if (data.getCompressionType() != Compress_None)
        return false;

    if (encodeJPEG(data.vector(), options))
    {
        data.setCompressionType(Compress_JPEG);
        return true;
    }

    return false;
}


bool virvo::decodeJPEG(CompressedVector& data, JPEGOptions& options)
{
    if (data.getCompressionType() != Compress_JPEG)
        return false;

    if (decodeJPEG(data.vector(), options))
    {
        data.setCompressionType(Compress_None);
        return true;
    }

    return false;
}
