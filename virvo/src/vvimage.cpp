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

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include <iostream>
#include "vvvirvo.h"
#include "vvimage.h"
#include "vvdebugmsg.h"
#if defined(VV_FFMPEG) || defined(VV_XVID)
#include "vvideo.h"
#endif

using namespace std;

//----------------------------------------------------------------------------
/** Constructor for initialization with an image
    @param h   picture height
    @param w   picture width
    @param image   pointer to the image
*/
vvImage::vvImage(short h, short w, uchar* image)
: height(h), width(w), imageptr(image)
{
  vvDebugMsg::msg(3, "vvImage::vvImage(): ", w, h);
  videosize = 0;
  size = height*width*4;
  codetype = 0;
  codedimage = new uchar[size];
  videoimageptr = new uchar[width*height*6];
  videocodedimage = new uchar[width*height*6];
  tmpimage = new uchar[width*height];
  t = VV_SERVER;
  videostyle = 0;
  videoquant = 1;
#if defined(VV_FFMPEG) || defined(VV_XVID)
  videoEncoder = new vvideo();
  if(videoEncoder->create_enc(w, h) == -1)
  {
    cerr << "vvImage::vvImage(): failed to create video encoder" << endl;
    delete videoEncoder;
    videoEncoder = NULL;
  }
  videoDecoder = new vvideo();
  if(videoDecoder->create_dec(w, h) == -1)
  {
    cerr << "vvImage::vvImage(): failed to create video decoder" << endl;
    delete videoDecoder;
    videoDecoder = NULL;
  }
#endif
}

//----------------------------------------------------------------------------
/** Constructor for an empty image
 */
vvImage::vvImage()
{
  height = 0;
  width = 0;
  size = 0;
  videosize = 0;
  imageptr = 0;
  codedimage = 0;
  videoimageptr = 0;
  videocodedimage = 0;
  tmpimage = 0;
  codetype = 0;
  t = VV_CLIENT;
  videostyle = 0;
  videoquant = 1;
#if defined(VV_FFMPEG) || defined(VV_XVID)
  videoEncoder = new vvideo();
  videoDecoder = new vvideo();
#endif
}

//----------------------------------------------------------------------------
/** Destructor
 */
vvImage::~vvImage()
{
  if (t == VV_CLIENT)
  {
    if (imageptr == codedimage)
    {
      if (imageptr != 0)
        delete[] imageptr;
    }
    else
    {
      if (imageptr != 0)
        delete[] imageptr;
      if (codedimage != 0)
        delete[] codedimage;
    }
  }
  else
    delete[] codedimage;
  if (videoimageptr !=0)
    delete[] videoimageptr;
  if (videocodedimage != 0)
    delete[] videocodedimage;
  if (tmpimage != 0)
    delete[] tmpimage;
#if defined(VV_FFMPEG) || defined(VV_XVID)
  if (videoEncoder != 0)
    delete videoEncoder;
  if (videoDecoder != 0)
    delete videoDecoder;
#endif
}

//----------------------------------------------------------------------------
/**Reinitializes an image object with new width and height
@param h   picture height
@param w   picture width
@param image   pointer to the image
*/
void vvImage::setNewImage(short h, short w, uchar* image)
{
  vvDebugMsg::msg(3, "vvImage::setNewImage(): ", w, h);
#if defined(VV_FFMPEG) || defined(VV_XVID)
  if (width != w || height != h)
  {
    if(videoEncoder->create_enc(w, h) == -1)
    {
      cerr << "vvImage::vvImage(): failed to create video encoder" << endl;
      delete videoEncoder;
      videoEncoder = NULL;
      return;
    }
    if(videoDecoder->create_dec(w, h) == -1)
    {
      cerr << "vvImage::vvImage(): failed to create video decoder" << endl;
      delete videoDecoder;
      videoDecoder = NULL;
      return;
    }
  }
#endif
  height =h;
  width = w;
  imageptr = image;
  size = height*width*4;
  codetype = 0;
  if (codedimage != 0)
    delete[] codedimage;
  codedimage = new uchar[size];
  if (videoimageptr !=0)
    delete [] videoimageptr;
  if (videocodedimage != 0)
    delete [] videocodedimage;
  if (tmpimage != 0)
    delete [] tmpimage;
  videoimageptr = new uchar[width*height*6];
  videocodedimage = new uchar[width*height*6];
  tmpimage = new uchar[width*height];
}

//----------------------------------------------------------------------------
/**Sets the image pointer to a new image which has the same height and
width as the old one.
@param image   pointer to the image
*/
void vvImage::setNewImagePtr(uchar* image)
{
  imageptr = image;
  size = height*width*4;
  codetype = 0;
}

//----------------------------------------------------------------------------
/**Encodes an image
@param ct   codetype to use (see detailed description of the class)
@param sw   start pixel relating to width
@param ew   end pixel relating to width
@param sh   start pixel relating to height
@param eh   end pixel relating to height
@return size of encoded image in bytes, or -1 on error
*/
int vvImage::encode(short ct, short sw, short ew, short sh, short eh)
{
  short realheight, realwidth;
  int start;
  float cr;

  if (size <= 0)
  {
    vvDebugMsg::msg(1, "Illegal image parameters ");
    return -1;
  }
  unsigned sum=0;
  for(int i=0; i<height; i++)
  {
    for(int j=0; j<width; j++)
    {
      sum += imageptr[(i*width+j)*4]+ imageptr[(i*width+j)*4+1]+ imageptr[(i*width+j)*4+2]+ imageptr[(i*width+j)*4+3];
    }
  }
  fprintf(stderr, "csum=0x%08x\n", sum);
  switch(ct)
  {
    case 0:cr=1;break;
    case 1:
    {
      if (spec_RLC_encode(0, height, width))
      {
        vvDebugMsg::msg(1, "No compression possible");
        codetype = 0;
      }
      else
        codetype = 1;
      cr = (float)size / (height*width*4);
    }break;
    case 2:
    {
      codetype = 2;
      if(sh<0 || eh<0  || sw<0 || ew<0 ||
        (realheight=short(eh-sh+1))<=0 || (realwidth=short(ew-sw+1))<=0 ||
        eh > height-1 || ew > width-1)
      {
        vvDebugMsg::msg(1,"Wrong usage vvImage::encode()");
        return -1;
      }
      start = (sh)*width*4 + (sw)*4;
      vvToolshed::write32(&codedimage[0],(ulong)start);
      vvToolshed::write16(&codedimage[4],realwidth);
      if (spec_RLC_encode(start, realheight, realwidth, 6))
      {
        vvDebugMsg::msg(1,"No compression possible");
        codetype = 0;
      }
      cr = (float)size / (height*width*4);
    }break;
#if defined(VV_FFMPEG) || defined(VV_XVID)
    case 3:
    {
      int i;
      codetype = 3;
      for (i=0; i<width*height; ++i)
        memcpy(&videoimageptr[i * 3], &imageptr[i * 4], 3);
      for (i=0; i<width*height; ++i)
        memcpy(&tmpimage[i], &imageptr[i * 4 +3], 1);
      imageptr = tmpimage;
      //memset(imageptr, 0xff, width*height*1);
      if (videoEncode())
      {
        vvDebugMsg::msg(1,"Error: videoEncode()");
        return -1;
      }
      if ( (size = gen_RLC_encode(imageptr, codedimage, width*height, 1, width*height*4)) < 0)
      {
        vvDebugMsg::msg(1,"Error: gen_RLC_encode()");
        return -1;
      }
      imageptr = codedimage;
      cr = (float)(size+videosize) / (height*width*4);
    }break;
#endif
    default:
      vvDebugMsg::msg(1,"Unknown encoding type ", ct );
      return -1;
  }
  vvDebugMsg::msg(2, "compression rate: ", cr);
  vvDebugMsg::msg(3, "image encoding succeeded");
  return size;
}

//----------------------------------------------------------------------------
/** Decodes an image
 */
int vvImage::decode()
{
  short  realwidth;
  int start;

  switch(codetype)
  {
    case 0: imageptr = codedimage;break;
    case 1:
    {
      spec_RLC_decode(0, width);
    }break;
    case 2:
    {
      memset(imageptr, 0, height*width*4);
      start = (int)vvToolshed::read32(&codedimage[0]);
      realwidth = vvToolshed::read16(&codedimage[4]);
      spec_RLC_decode(start, realwidth, 6);
    }break;
#if defined(VV_FFMPEG) || defined(VV_XVID)
    case 3:
    {
      int i;
      if (videoDecode())
      {
        vvDebugMsg::msg(1,"Error: videoDecode()");
        return -1;
      }
      for (i=0; i<width*height; ++i)
        memcpy(&imageptr[i * 4], &videoimageptr[i * 3], 3);
      if (gen_RLC_decode(codedimage, tmpimage, size, 1, width*height))
      {
        vvDebugMsg::msg(1,"Error: gen_RLC_decode()");
        return -1;
      }
      for (i=0; i<width*height; ++i)
        memcpy(&imageptr[i * 4 + 3], &tmpimage[i], 1);
      size = width*height*4;
    }break;
#endif
    default:
      vvDebugMsg::msg(1,"No encoding type with that identifier");
      return -1;
  }
  codetype = 0;
  vvDebugMsg::msg(3, "image decoding succeeded");
  return 0;
}

//----------------------------------------------------------------------------
/** Sets the image height.
 */
void vvImage::setHeight(short h)
{
  height = h;
}

//----------------------------------------------------------------------------
/** Sets the image width.
 */
void vvImage::setWidth(short w)
{
  width = w;
}

//----------------------------------------------------------------------------
/** Sets the code type.
 */
void vvImage::setCodeType(short ct)
{
  codetype = ct;
}

//----------------------------------------------------------------------------
/** Sets the image size.
 */
void vvImage::setSize(int s)
{
  size = s;
}

//----------------------------------------------------------------------------
/** Sets the video image size.
 */
void vvImage::setVideoSize(int s)
{
  fprintf(stderr, "setVideoSize: s=%d\n", s);
  videosize = s;
}

//----------------------------------------------------------------------------
/** Sets the image pointer.
 */
void vvImage::setImagePtr(uchar* image)
{
  imageptr = image;
}

//----------------------------------------------------------------------------
/** Sets the style of video encoding
 */
void vvImage::setVideoStyle(int s)
{
  if ( (s<0) || (s>6) )
  {
    vvDebugMsg::msg(1, "videoStyle hast to be between 0 and 6, using 0 now");
    videostyle = 0;
  }
  else
    videostyle = s;
}

//----------------------------------------------------------------------------
/** Sets the value for the video quantizer
 */
void vvImage::setVideoQuant(int q)
{
  if ( (q<1) || (q>31) )
  {
    vvDebugMsg::msg(1,"videoQuant has to be between 1 and 31, using 1 now");
    videoquant = 1;
  }
  else
    videoquant = q;
}

//----------------------------------------------------------------------------
/** Sets an key frame
@param k
*/
void vvImage::setKeyframe(int k)
{
  keyframe = k;
}

//----------------------------------------------------------------------------
/**Returns the image height
 */
short vvImage::getHeight()
{
  return height;
}

//----------------------------------------------------------------------------
/** Returns the image width
 */
short vvImage::getWidth()
{
  return width;
}

//----------------------------------------------------------------------------
/** Returns the code type
 */
short vvImage::getCodeType()
{
  return codetype;
}

//----------------------------------------------------------------------------
/** Returns the image size in bytes
 */
int vvImage::getSize()
{
  return size;
}

//----------------------------------------------------------------------------
/** Returns the video image size in bytes
 */
int vvImage::getVideoSize()
{
  return videosize;
}

//----------------------------------------------------------------------------
/** Returns the key frame
 */
int vvImage::getKeyframe()
{
  return keyframe;
}

//----------------------------------------------------------------------------
/** Returns the pointer to the image
 */
uchar* vvImage::getImagePtr()
{
  return imageptr;
}

//----------------------------------------------------------------------------
/** Returns the pointer to the encoded image
 */
uchar* vvImage::getCodedImage()
{
  return codedimage;
}

//----------------------------------------------------------------------------
/** Returns the pointer to the encoded video image
 */
uchar* vvImage::getVideoCodedImage()
{
  return videocodedimage;
}

//----------------------------------------------------------------------------
/**Does the Run Length Encoding for a defined cutout of an image.
@param start   start pixel for RLE encoding
@param h   height of pixel square to encode
@param w   width of pixel square to encode
@param dest   start writing in coded image at position dest
*/
int vvImage::spec_RLC_encode(int start, short h, short w, int dest)
{
  short samePixel=1;
  short diffPixel=0;
  int src;
  int l,m;

  for ( int i=0; i < h; i++)
  {
    src = start + i*width*4;
    for ( int j=0; j < w; j++)
    {
      if (j == (w-1))
      {
        l=1;
        if (i == (h-1))
          m=0;
        else
          m =1;
      }
      else
      {
        m=1;
        l=0;
      }
      if (imageptr[src] == imageptr[m*(src+4+l*(width-w)*4)] &&
        imageptr[src+1] == imageptr[m*(src+5+l*(width-w)*4)] &&
        imageptr[src+2] == imageptr[m*(src+6+l*(width-w)*4)] &&
        imageptr[src+3] == imageptr[m*(src+7+l*(width-w)*4)] )
      {
        if(samePixel == 129)
          put_same(samePixel, dest);
        else
        {
          samePixel++;
          if(diffPixel > 0 )
            put_diff(diffPixel, dest);
          if(samePixel == 2)
          {
            if ((dest+5) > size)
              return -1;
            memcpy(&codedimage[dest+1], &imageptr[src], 4);
          }
        }
      }
      else
      {
        if (samePixel > 1)
          put_same(samePixel, dest);
        else
        {
          if ((dest+5+4*diffPixel) > size)
            return -1;
          memcpy(&codedimage[dest+1+diffPixel*4], &imageptr[src], 4);
          diffPixel++;
          if(diffPixel == 128)
            put_diff(diffPixel, dest);
        }
      }
      src += 4;
    }
  }
  if (samePixel > 1)
  {
    samePixel--;
    put_same(samePixel, dest);
  }
  else if (diffPixel > 0)
    put_diff(diffPixel, dest);
  imageptr = codedimage;
  size = dest;
  return 0;
}

//----------------------------------------------------------------------------
/** Does the Run Length Decoding for a cutout of an image
@param start   start pixel where the decoded pixel square is
written
@param w   width of pixel square to decode
@param src   start position of encoded pixels in coded image
*/
int vvImage::spec_RLC_decode(int start, short w, int src)
{
  int dest;
  int length;

  dest = start;
  while (src < size)
  {
    length = (int)codedimage[src];
    if (length > 127)
    {
      for(int i=0; i<(length - 126); i++)
      {
        if (((dest-start-4*w)% (4*width)) == 0 && dest != start)
          dest += (width-w)*4;
        memcpy(&imageptr[dest], &codedimage[src+1], 4);
        dest += 4;
      }
      src += 5;
    }
    else
    {
      length++;
      for(int i=0; i<(length); i++)
      {
        if (((dest-start-4*w)% (4*width)) == 0 && dest != start)
          dest += (width-w)*4;
        memcpy(&imageptr[dest], &codedimage[src+1+i*4], 4);
        dest +=4;
      }
      src += 1+4*length;
    }
  }
  size = height*width*4;
  return 0;
}

//----------------------------------------------------------------------------
/** Writes a RLE encoded set of same pixels.
@param sP   number of same pixels
@param d   destination in coded image where to write
*/
void vvImage::put_same(short& sP, int& d)
{
  codedimage[d] = (uchar)(126+sP);
  d += 5;
  sP = 1;
}

//----------------------------------------------------------------------------
/** Writes a RLE encoded set of different pixels.
@param dP   number of different pixels
@param d   destination in coded image where to write
*/
void vvImage::put_diff(short& dP, int& d)
{
  codedimage[d] = (uchar)(dP-1);
  d += 1+4*dP;
  dP=0;
}

//----------------------------------------------------------------------------
/** Allocates momory for a new image
 */
int vvImage::alloc_mem()
{
  vvDebugMsg::msg(3, "vvImage::alloc_mem(): ", width, height);

#if defined(VV_FFMPEG) || defined(VV_XVID)
  if(videoEncoder->create_enc(width, height) == -1)
  {
    cerr << "vvImage::vvImage(): failed to create video encoder" << endl;
    delete videoEncoder;
    videoEncoder = NULL;
    return -1;
  }
  if(videoDecoder->create_dec(width, height) == -1)
  {
    cerr << "vvImage::vvImage(): failed to create video decoder" << endl;
    delete videoDecoder;
    videoDecoder = NULL;
    return -1;
  }
#endif

  if (imageptr == codedimage)
  {
    if (imageptr != 0)
      delete[] imageptr;
  }
  else
  {
    if (imageptr != 0)
      delete[] imageptr;
    if (codedimage != 0)
      delete[] codedimage;
  }
  if (videoimageptr !=0)
    delete [] videoimageptr;
  if (videocodedimage != 0)
    delete [] videocodedimage;
  if (tmpimage != 0)
    delete [] tmpimage;
  if (codetype != 0)
  {
    imageptr = new uchar[height*width*4];
    if (!imageptr)
      return -1;
  }
  if (codetype == 3)
  {
    videoimageptr = new uchar[height*width*6];
    if (!videoimageptr)
      return -1;
    videocodedimage = new uchar[height*width*6];
    if (!videocodedimage)
      return -1;
    tmpimage = new uchar[height*width];
    if (!tmpimage)
      return -1;
  }
  codedimage = new uchar[height*width*4];
  if (!codedimage)
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
/** Does the video encoding
 */
int vvImage::videoEncode()
{
#if defined(VV_FFMPEG) || defined(VV_XVID)
  assert(videoEncoder);
  videosize = width * height * 6;
  fprintf(stderr, "calling videoEncoder->enc_frame\n");
  if (videoEncoder->enc_frame(videoimageptr, videocodedimage, &videosize, &keyframe))
  {
    vvDebugMsg::msg(1,"Error videoEncode()");
    return -1;
  }
  vvDebugMsg::msg(3, "encoded video image size: ", videosize);
#else
  vvDebugMsg::msg(1, "vvImage::videoEncode(): doing nothing");
#endif
  return 0;
}

//----------------------------------------------------------------------------
/** Does the video decoding
 */
int vvImage::videoDecode()
{
#if defined(VV_FFMPEG) || defined(VV_XVID)
  int newsize = width * height * 6;

  assert(videoDecoder);
  if (videoDecoder->dec_frame(videocodedimage, videoimageptr, videosize, &newsize))
  {
    vvDebugMsg::msg(1,"Error videoDecode()");
    return -1;
  }
#else
  vvDebugMsg::msg(1, "vvImage::videoDecode(): doing nothing");
#endif
  return 0;
}

//----------------------------------------------------------------------------
/** general function for the RLC encoding
 */
                                                  // size=total size in byte
int vvImage::gen_RLC_encode(uchar* in, uchar* out, int size, int symbol_size, int space)
{
  int same_symbol=1;
  int diff_symbol=0;
  int src=0;
  int dest=0;
  bool same;
  int i;

  if ((size % symbol_size) != 0)
  {
    vvDebugMsg::msg(1,"No RLC encoding possible with this parameters");
    return -1;
  }

  while (src < (size - symbol_size))
  {
    same = true;
    for (i=0; i<symbol_size; i++)
    {
      if (in[src+i] != in[src+symbol_size+i])
      {
        same = false;
        break;
      }
    }
    if (same)
    {
      if (same_symbol == 129)
      {
        out[dest] = (uchar)(126+same_symbol);
        dest += symbol_size+1;
        same_symbol = 1;
      }
      else
      {
        same_symbol++;
        if (diff_symbol > 0)
        {
          out[dest] = (uchar)(diff_symbol-1);
          dest += 1+symbol_size*diff_symbol;
          diff_symbol=0;
        }
        if (same_symbol == 2)
        {
          if ((dest+1+symbol_size) > space)
          {
            vvDebugMsg::msg(1,"Not enough memory to encode");
            return -1;
          }
          memcpy(&out[dest+1], &in[src], symbol_size);
        }
      }
    }
    else
    {
      if (same_symbol > 1)
      {
        out[dest] = (uchar)(126+same_symbol);
        dest += symbol_size+1;
        same_symbol = 1;
      }
      else
      {
        if ((dest+1+diff_symbol*symbol_size+symbol_size) > space)
        {
          vvDebugMsg::msg(1,"Not enough memory to encode");
          return -1;
        }
        memcpy(&out[dest+1+diff_symbol*symbol_size], &in[src], symbol_size);
        diff_symbol++;
        if (diff_symbol == 128)
        {
          out[dest] = (uchar)(diff_symbol-1);
          dest += 1+symbol_size*diff_symbol;
          diff_symbol=0;
        }
      }
    }
    src += symbol_size;
  }
  if (same_symbol > 1)
  {
    out[dest] = (uchar)(126+same_symbol);
    dest += symbol_size+1;
  }
  else
  {
    if ((dest+1+diff_symbol*symbol_size+symbol_size) > space)
    {
      vvDebugMsg::msg(1,"Not enough memory to encode");
      return -1;
    }
    memcpy(&out[dest+1+diff_symbol*symbol_size], &in[src], symbol_size);
    diff_symbol++;
    out[dest] = (uchar)(diff_symbol-1);
    dest += 1+symbol_size*diff_symbol;
  }
  if (dest > size)
  {
    vvDebugMsg::msg(1,"No compression possible with RLC !!!");
  }
  return dest;
}

//----------------------------------------------------------------------------
/** general function for the RLC decoding
 */
int vvImage::gen_RLC_decode(uchar* in, uchar* out, int size, int symbol_size, int space)
{
  int src=0;
  int dest=0;
  int i, length;

  while (src < size)
  {
    length = (int)in[src];
    if (length > 127)
    {
      for(i=0; i<(length - 126); i++)
      {
        if ((dest + symbol_size) > space)
        {
          vvDebugMsg::msg(1,"Not enough memory to decode");
          return -1;
        }
        memcpy(&out[dest], &in[src+1], symbol_size);
        dest += symbol_size;
      }
      src += 1+symbol_size;
    }
    else
    {
      length++;
      if ((dest + length*symbol_size) > space)
      {
        vvDebugMsg::msg(1,"Not enough memory to decode");
        return -1;
      }
      memcpy(&out[dest], &in[src+1], symbol_size*length);
      dest += length*symbol_size;
      src += 1+symbol_size*length;
    }
  }
  return 0;
}
