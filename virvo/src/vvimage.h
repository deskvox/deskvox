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

#ifndef _VVIMAGE_H_
#define _VVIMAGE_H_

#include <string.h>
#include <assert.h>

#include "vvexport.h"
#include "vvtoolshed.h"
#if defined(VV_FFMPEG) || defined(VV_XVID)
#include "vvideo.h"
#endif

//----------------------------------------------------------------------------
/**This class provides different encoding and decoding types for RGB images. <BR>

Supported code types:
- no encoding (code type 0)
- Run Length Encoding over the whole image (code type 1)
- Run Length Encoding over a quadratic part of the image (code type 2).
  Therefore start and end pixels for width an height must be specified.
  The rest of the image is interpreted as background and the pixels get the
  value 0,0,0,0. (picture width from 0 - width-1, picture height from 0 - height-1)
- Video Encoding (code type 3). For this type the VV_FFMPEG/VV_XVID Flag must be set.<BR>

Here is an example code fragment for encoding and decoding an image with
800 x 600 pixels :<BR>
<PRE>

//Create a new image class instance
vvImage* im = new vvImage(600, 800, (char *)imagepointer);

//Encode with normal RLE
if(im->encode(1) < 0)
{
delete im;
return -1;
}

//Or encode with RLE but only the lower half of the image
if(im->encode(2, 0, 799, 300, 599 ) < 0)
{
delete im;
return -1;
}
//Decode the image
if(im->decode())
return -1;

delete im;
</PRE>
*/
class VIRVOEXPORT vvImage
{
  public:

    vvImage(short, short, uchar*);
    vvImage();
    virtual ~vvImage();
    int encode(short, short sh=-1, short eh=-1, short sw=-1, short ew=-1);
    int decode();
    void setNewImage(short, short, uchar*);
    void setHeight(short);
    void setWidth(short);
    void setCodeType(short);
    void setSize(int);
    void setVideoSize(int);
    void setImagePtr(uchar*);
    void setKeyframe(int);
    void setNewImagePtr(uchar*);
    void setVideoStyle(int);
    void setVideoQuant(int);
    short getHeight();
    short getWidth();
    short getCodeType();
    int getSize();
    int getVideoSize();
    int getKeyframe();
    uchar* getImagePtr();
    uchar* getCodedImage();
    uchar* getVideoCodedImage();
    int alloc_mem();

  private:

    enum Type
    {
      VV_SERVER,
      VV_CLIENT
    };

    Type t;
    short height;
    short width;
    short codetype;
    int size;
    int videosize;
    int keyframe;
    uchar* imageptr;
    uchar* codedimage;
    uchar* videoimageptr;
    uchar* videocodedimage;
    uchar* tmpimage;
    int videostyle;
    int videoquant;
#if defined(VV_FFMPEG) || defined(VV_XVID)
    vvideo* videoEncoder;
    vvideo* videoDecoder;
#endif

    int spec_RLC_encode(int, short, short, int dest=0);
    int spec_RLC_decode(int, short, int src=0);
    int gen_RLC_encode(uchar*, uchar*, int, int, int);
    int gen_RLC_decode(uchar*, uchar*, int, int, int);
    void put_diff(short&, int&);
    void put_same(short&, int&);
    int videoEncode();
    int videoDecode();
};
#endif
