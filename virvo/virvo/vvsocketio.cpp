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

#include "vvsocketio.h"
#include "vvibrimage.h"
#include "vvvoldesc.h"
#include "vvdebugmsg.h"
#include "vvbrick.h"
#include "vvmulticast.h"
#include "vvtoolshed.h"

//#ifdef VV_DEBUG_MEMORY
//#include <crtdbg.h>
//#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
//#endif

#include <sstream>
#include <string>

//----------------------------------------------------------------------------
/** Constructor
 @param sock ready to use socket of type vvSocket
*/
vvSocketIO::vvSocketIO(vvSocket *sock)
: _socket(sock)
{
}

//----------------------------------------------------------------------------
/// Destructor
vvSocketIO::~vvSocketIO()
{
}

//----------------------------------------------------------------------------
/** Checks if there is data in the socket receive buffer.
 @return  true for data in the socket receive buffer, false for not.
*/
bool vvSocketIO::sock_action()
{
  if(_socket)
  {
    if (_socket->isDataWaiting() > 0)
      return true;
    else
      return false;
  }
  else
    return false;
}

//----------------------------------------------------------------------------
/** Get volume attributes from socket.
  @param vd  empty volume description which is to be filled with the volume attributes
*/
vvSocket::ErrorType vvSocketIO::getVolumeAttributes(vvVolDesc* vd)
{
  if(_socket)
  {
    vvSocket::ErrorType retval;

    int size = vd->serializeAttributes();

    std::vector<uchar> buffer(size+4);
    if ((retval =_socket->readData(&buffer[0], size+4)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Header received");
    vd->deserializeAttributes(&buffer[0]);
    vd->_scale = vvToolshed::readFloat(&buffer[size]);

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}


//----------------------------------------------------------------------------
/** Get volume data from socket.
  @param vd  empty volume description which is to be filled with the volume data
*/
vvSocket::ErrorType vvSocketIO::getVolume(vvVolDesc* vd, vvMulticastParameters *mcParam)
{
  if(_socket)
  {
    vvSocket::ErrorType retval = getVolumeAttributes(vd);
    if(retval != vvSocket::VV_OK)
      return retval;

    int size = vd->getFrameBytes();
    bool tryMC = false;
    getBool(tryMC);

    if(tryMC)
    {
      vvMulticast mcSock = mcParam
                         ? vvMulticast(vvMulticast::VV_RECEIVER, mcParam->api, mcParam->addr, mcParam->port)
                         : vvMulticast(vvMulticast::VV_RECEIVER);
      putBool(true);
      for(int k =0; k< vd->frames; k++)
      {
        uchar *buffer = new uchar[size];
        if (!buffer)
          return vvSocket::VV_ALLOC_ERROR;
        if (mcSock.read(buffer, size, 3.0) != size) // set timeout!
        {
          delete[] buffer;
          putBool(false);
          goto tcpTransfer;
        }
        vd->addFrame(buffer, vvVolDesc::ARRAY_DELETE);
      }
      putBool(true);
      return vvSocket::VV_OK;
    }

    tcpTransfer:
    for(int k =0; k< vd->frames; k++)
    {
      uchar *buffer = new uchar[size];
      if (!buffer)
        return vvSocket::VV_ALLOC_ERROR;
      if ((retval =_socket->readData(buffer, size)) != vvSocket::VV_OK)
      {
        delete[] buffer;
        return retval;
      }
      vd->addFrame(buffer, vvVolDesc::ARRAY_DELETE);
    }
    vvDebugMsg::msg(3, "Data received");
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write volume attributes to socket.
  @param vd  volume description of volume to be send.
*/
vvSocket::ErrorType vvSocketIO::putVolumeAttributes(const vvVolDesc* vd)
{
  if(_socket)
  {
    int size = vd->serializeAttributes();
    std::vector<uchar> buffer(size+4);
    vd->serializeAttributes(&buffer[0]);
    vvToolshed::writeFloat(&buffer[size], vd->_scale);
    vvDebugMsg::msg(3, "Sending header ...");
    return _socket->writeData(&buffer[0], size+4);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write volume data to socket.
  @param vd  volume description of volume to be send.
*/
vvSocket::ErrorType vvSocketIO::putVolume(const vvVolDesc* vd, bool tryMC, bool mcMaster, vvMulticastParameters *mcParam)
{
  if(_socket)
  {
    vvSocket::ErrorType retval = putVolumeAttributes(vd);
    if(retval != vvSocket::VV_OK)
      return retval;

    int frames = vd->frames;

    int size = vd->getFrameBytes();
    vvDebugMsg::msg(3, "Sending data ...");

    putBool(tryMC);
    if(tryMC)
    {
      if(mcMaster)
      {
        for(int k=0; k < frames; k++)
        {
          vvMulticast mcSock = mcParam
                               ? vvMulticast(vvMulticast::VV_SENDER, mcParam->api, mcParam->addr, mcParam->port)
                               : vvMulticast(vvMulticast::VV_SENDER);

          const uchar *buffer = vd->getRaw(k);
          bool ready;
          getBool(ready);
          if(!ready) break; // unexpected answer

          if (mcSock.write(buffer, size, 3.0) != size) // set timeout!
          {
            return vvSocket::VV_WRITE_ERROR;
          }
        }
      }
      bool mcAnswer = false;
      getBool(mcAnswer);
      if(mcAnswer)
      {
        return vvSocket::VV_OK;
      }
      else
      {
        vvDebugMsg::msg(3, "vvSocketIO::putVolume() Multicast-transfer failed. Fallback: sending via TCP...");
      }
    }

    for(int k=0; k < frames; k++)
    {
      const uchar *buffer = vd->getRaw(k);
      if ((retval =_socket->writeData(buffer, size)) != vvSocket::VV_OK)
      {
        return retval;
      }
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Get a transfer function from the socket.
  @param tf  pointer to a vvTransFunc.
*/
vvSocket::ErrorType vvSocketIO::getTransferFunction(vvTransFunc& tf)
{
  if(_socket)
  {
    uchar* buffer = NULL;
    vvSocket::ErrorType retval;
    int len;

    if ((retval = getInt32(len)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }

    buffer = new uchar[len+1];
    if ((retval =_socket->readData(buffer, len)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    buffer[len] = '\0';

    std::istringstream in;
    in.str((const char*)buffer);
  #ifdef WIN32
    char cline[65535];
  #else
    char cline[vvTFWidget::MAX_STR_LEN];
  #endif
    while (in.getline(cline, vvTFWidget::MAX_STR_LEN))
    {
      std::string line = std::string(cline);

      // Skip over erroneous lines.
      if (line.length() < 3)
      {
        continue;
      }

      std::vector<std::string> tokens = vvToolshed::split(line, " ");

      // At least widget type and name.
      if (tokens.size() < 2)
      {
        continue;
      }
      const char* name = tokens[0].c_str();

      vvTFWidget* widget = vvTFWidget::produce(vvTFWidget::getWidgetType(name));

      if (widget)
      {
        widget->fromString(line);
        tf._widgets.push_back(widget);
      }
    }

    delete[] buffer;
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write a transfer function to the socket.
  @param tf  pointer to a vvTransFunc.
*/
vvSocket::ErrorType vvSocketIO::putTransferFunction(vvTransFunc& tf)
{
  if(_socket)
  {
    uchar* buffer = NULL;
    vvSocket::ErrorType retval;

    std::ostringstream out;

    for (std::vector<vvTFWidget*>::const_iterator it = tf._widgets.begin();
         it != tf._widgets.end(); ++it)
    {
      out << (*it)->toString();
    }

    const size_t len = strlen(out.str().c_str());
    buffer = new uchar[len+1];
    strcpy((char*)buffer, out.str().c_str());

    putInt32((int)len);

    if ((retval =_socket->writeData(buffer, len)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }

    delete[] buffer;
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Get a single brick from the socket.
  @param brick  pointer to a vvBrick.
*/
vvSocket::ErrorType vvSocketIO::getBrick(vvBrick* brick)
{
  if(_socket)
  {
    uchar* buffer;
    vvSocket::ErrorType retval;

    const int sob = sizeOfBrick();
    buffer = new uchar[sob];

    if ((retval =_socket->readData(buffer, sob)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }

    brick->pos[0] = vvToolshed::readFloat(&buffer[0]);
    brick->pos[1] = vvToolshed::readFloat(&buffer[4]);
    brick->pos[2] = vvToolshed::readFloat(&buffer[8]);

    brick->min[0] = vvToolshed::readFloat(&buffer[12]);
    brick->min[1] = vvToolshed::readFloat(&buffer[16]);
    brick->min[2] = vvToolshed::readFloat(&buffer[20]);

    brick->max[0] = vvToolshed::readFloat(&buffer[24]);
    brick->max[1] = vvToolshed::readFloat(&buffer[28]);
    brick->max[2] = vvToolshed::readFloat(&buffer[32]);

    brick->minValue = (int)vvToolshed::read32(&buffer[36]);
    brick->maxValue = (int)vvToolshed::read32(&buffer[40]);

    brick->visible = (buffer[44] != 0);
    brick->insideProbe = (buffer[46] != 0);
    // One byte for padding.
    brick->index = (int)vvToolshed::read32(&buffer[48]);

    brick->startOffset[0] = (int)vvToolshed::read32(&buffer[52]);
    brick->startOffset[1] = (int)vvToolshed::read32(&buffer[56]);
    brick->startOffset[2] = (int)vvToolshed::read32(&buffer[60]);

    brick->texels[0] = (int)vvToolshed::read32(&buffer[64]);
    brick->texels[1] = (int)vvToolshed::read32(&buffer[68]);
    brick->texels[2] = (int)vvToolshed::read32(&buffer[72]);

    brick->dist = vvToolshed::readFloat(&buffer[76]);

    brick->texRange[0] = vvToolshed::readFloat(&buffer[80]);
    brick->texRange[1] = vvToolshed::readFloat(&buffer[84]);
    brick->texRange[2] = vvToolshed::readFloat(&buffer[88]);

    brick->texMin[0] = vvToolshed::readFloat(&buffer[92]);
    brick->texMin[1] = vvToolshed::readFloat(&buffer[96]);
    brick->texMin[2] = vvToolshed::readFloat(&buffer[100]);

    delete[] buffer;

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write a single brick to the socket.
  @param brick  pointer to a vvBrick.
*/
vvSocket::ErrorType vvSocketIO::putBrick(const vvBrick* brick)
{
  if(_socket)
  {
    uchar* buffer;
    vvSocket::ErrorType retval;

    const int sob = sizeOfBrick();
    buffer = new uchar[sob];

    vvToolshed::writeFloat(&buffer[0], brick->pos[0]);
    vvToolshed::writeFloat(&buffer[4], brick->pos[1]);
    vvToolshed::writeFloat(&buffer[8], brick->pos[2]);

    vvToolshed::writeFloat(&buffer[12], brick->min[0]);
    vvToolshed::writeFloat(&buffer[16], brick->min[1]);
    vvToolshed::writeFloat(&buffer[20], brick->min[2]);

    vvToolshed::writeFloat(&buffer[24], brick->max[0]);
    vvToolshed::writeFloat(&buffer[28], brick->max[1]);
    vvToolshed::writeFloat(&buffer[32], brick->max[2]);

    vvToolshed::write32(&buffer[36], brick->minValue);
    vvToolshed::write32(&buffer[40], brick->maxValue);

    buffer[44] = (uchar)brick->visible;
    buffer[46] = (uchar)brick->insideProbe;
    // One byte for padding.
    vvToolshed::write32(&buffer[48], brick->index);

    vvToolshed::write32(&buffer[52], brick->startOffset[0]);
    vvToolshed::write32(&buffer[56], brick->startOffset[1]);
    vvToolshed::write32(&buffer[60], brick->startOffset[2]);

    vvToolshed::write32(&buffer[64], brick->texels[0]);
    vvToolshed::write32(&buffer[68], brick->texels[1]);
    vvToolshed::write32(&buffer[72], brick->texels[2]);

    vvToolshed::writeFloat(&buffer[76], brick->dist);

    vvToolshed::writeFloat(&buffer[80], brick->texRange[0]);
    vvToolshed::writeFloat(&buffer[84], brick->texRange[1]);
    vvToolshed::writeFloat(&buffer[88], brick->texRange[2]);

    vvToolshed::writeFloat(&buffer[92], brick->texMin[0]);
    vvToolshed::writeFloat(&buffer[96], brick->texMin[1]);
    vvToolshed::writeFloat(&buffer[100], brick->texMin[2]);

    if ((retval =_socket->writeData(buffer, sob)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }

    delete[] buffer;

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Get brick list from the socket. Bricks contain no volume data.
  @param bricks  std::vector with pointers to bricks.
*/
vvSocket::ErrorType vvSocketIO::getBricks(std::vector<vvBrick*>& bricks)
{
  if(_socket)
  {
    uchar* buffer;
    vvSocket::ErrorType retval;

    buffer = new uchar[4];

    if ((retval =_socket->readData(buffer, 4)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    const int numBricks = vvToolshed::read32(&buffer[0]);
    delete[] buffer;

    bricks.resize(numBricks);

    for (int i=0; i<numBricks; ++i)
    {
      vvBrick* brick = new vvBrick();
      if ((retval = getBrick(brick)) != vvSocket::VV_OK)
      {
        return retval;
      }
      bricks[i] = brick;
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write brick list to the socket. Bricks contain no volume data.
  @param bricks  std::vector with pointers to bricks.
*/
vvSocket::ErrorType vvSocketIO::putBricks(const std::vector<vvBrick*>& bricks)
{
  if(_socket)
  {
    uchar* buffer;
    vvSocket::ErrorType retval;

    const int numBricks = (const int)bricks.size();

    buffer = new uchar[4];
    vvToolshed::write32(&buffer[0], numBricks);

    vvDebugMsg::msg(3, "Sending num bricks ...");
    if ((retval =_socket->writeData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    delete[] buffer;

    for(std::vector<vvBrick*>::const_iterator it = bricks.begin(); it != bricks.end(); ++it)
    {
      vvBrick* brick = (*it);
      if ((retval = putBrick(brick)) != vvSocket::VV_OK)
      {
        return retval;
      }
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Get an image from the socket.
 @param im  pointer to a vvImage object.
*/
vvSocket::ErrorType vvSocketIO::getImage(vvImage* im)
{
  if(_socket)
  {
    const int BUFSIZE = 13;
    uchar buffer[BUFSIZE];
    vvSocket::ErrorType retval;
    short w, h;
    int imagesize;
    int videosize;

    if ((retval =_socket->readData(&buffer[0], BUFSIZE)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Header received");
    w = vvToolshed::read16(&buffer[2]);
    h = vvToolshed::read16(&buffer[0]);

    vvImage::CodeType ct = (vvImage::CodeType)vvToolshed::read8(&buffer[4]);
    if (h != im->getHeight() || w  != im->getWidth() || ct != im->getCodeType() )
    {
      im->setCodeType(ct);
      im->setHeight(h);
      im->setWidth(w);
      if(im->alloc_mem())
        return vvSocket::VV_ALLOC_ERROR;
    }
    imagesize = (int)vvToolshed::read32(&buffer[5]);
    videosize = (int)vvToolshed::read32(&buffer[9]);
    im->setSize(imagesize);
    im->setVideoSize(videosize);
    if (vvDebugMsg::isActive(3))
      fprintf(stderr, "imgsize=%d, videosize=%d\n", imagesize, videosize);
    if ((retval =_socket->readData(im->getCodedImage(), imagesize)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Image data received");
    if (ct == vvImage::VV_VIDEO)
    {
      if ((retval =_socket->readData(im->getVideoCodedImage(), videosize)) != vvSocket::VV_OK)
      {
        return retval;
      }
      vvDebugMsg::msg(3, "Video Image data received");
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write an image to the socket.
 @param im  pointer to an vvImage object.
*/
vvSocket::ErrorType vvSocketIO::putImage(const vvImage* im)
{
  if(_socket)
  {
    const int BUFSIZE = 13;
    uchar buffer[BUFSIZE];
    vvSocket::ErrorType retval;
    int imagesize;
    int videosize;
    int ct;
    imagesize = im->getSize();
    videosize = im->getVideoSize();
    ct = im->getCodeType();
    vvToolshed::write16(&buffer[0], im->getHeight());
    vvToolshed::write16(&buffer[2], im->getWidth());
    vvToolshed::write8(&buffer[4], (uchar)ct);
    vvToolshed::write32(&buffer[5], (ulong)imagesize);
    vvToolshed::write32(&buffer[9], (ulong)videosize);

    vvDebugMsg::msg(3, "Sending header ...");
    if ((retval =_socket->writeData(&buffer[0], BUFSIZE)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Sending image data ...");
    if ((retval =_socket->writeData(im->getImagePtr(), imagesize)) != vvSocket::VV_OK)
    {
      return retval;
    }
    if (ct == vvImage::VV_VIDEO)
    {
      vvDebugMsg::msg(3, "Sending video image data ...");
      if ((retval =_socket->writeData(im->getVideoCodedImage(), videosize)) != vvSocket::VV_OK)
      {
        return retval;
      }
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Get an 2.5d-image from the socket.
 @param im  pointer to a vvImage2_5d object.
*/
vvSocket::ErrorType vvSocketIO::getIbrImage(vvIbrImage* im)
{
  if(_socket)
  {
    vvSocket::ErrorType err = getImage(im);
    if(err != vvSocket::VV_OK)
      return err;

    // Get modelview / projection matrix for the current frame
    vvMatrix pm;
    err = getMatrix(&pm);
    if(err != vvSocket::VV_OK)
      return err;
    im->setProjectionMatrix(pm);

    vvMatrix mv;
    err = getMatrix(&mv);
    if(err != vvSocket::VV_OK)
      return err;
    im->setModelViewMatrix(mv);

    vvGLTools::Viewport vp;
    err = getViewport(vp);
    if(err != vvSocket::VV_OK)
      return err;
    im->setViewport(vp);

    float drMin = 0.f, drMax = 0.f;
    err = getFloat(drMin);
    if(err != vvSocket::VV_OK)
      return err;
    err = getFloat(drMax);
    if(err != vvSocket::VV_OK)
      return err;
    im->setDepthRange(drMin, drMax);

    int dp;
    err = getInt32(dp);
    if(err != vvSocket::VV_OK)
      return err;
    im->setDepthPrecision(dp);

    int ct;
    err = getInt32(ct);
    if(err != vvSocket::VV_OK)
      return err;
    im->setDepthCodetype((vvImage::CodeType)ct);

    im->alloc_pd();

    int size;
    err = getInt32(size);
    if(err != vvSocket::VV_OK)
      return err;
    im->setDepthSize(size);

    err = getData(im->getCodedDepth(), size, vvSocketIO::VV_UCHAR);
    if(err != vvSocket::VV_OK)
      return err;

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write an 2.5d-image to the socket.
 @param im  pointer to an vvImage2_5d object.
*/
vvSocket::ErrorType vvSocketIO::putIbrImage(const vvIbrImage* im)
{
  if(_socket)
  {
    vvSocket::ErrorType err = putImage(im);
    if (err != vvSocket::VV_OK)
    {
      return err;
    }

    vvMatrix pm = im->getProjectionMatrix();
    err = putMatrix(&pm);
    if (err != vvSocket::VV_OK)
    {
      return err;
    }

    vvMatrix mv = im->getModelViewMatrix();
    err = putMatrix(&mv);
    if (err != vvSocket::VV_OK)
    {
      return err;
    }

    vvGLTools::Viewport vp = im->getViewport();
    err = putViewport(vp);
    if (err != vvSocket::VV_OK)
    {
      return err;
    }

    float drMin = 0.f, drMax = 0.f;
    im->getDepthRange(&drMin, &drMax);
    err = putFloat(drMin);
    if (err != vvSocket::VV_OK)
    {
      return err;
    }
    err = putFloat(drMax);
    if (err != vvSocket::VV_OK)
    {
      return err;
    }

    err = putInt32(im->getDepthPrecision());
    if (err != vvSocket::VV_OK)
    {
      return err;
    }

    err = putInt32(im->getDepthCodetype());
    if (err != vvSocket::VV_OK)
    {
      return err;
    }

    err = putInt32(im->getDepthSize());
    if (err != vvSocket::VV_OK)
    {
      return err;
    }

    err = putData(im->getCodedDepth(), im->getDepthSize(), vvSocketIO::VV_UCHAR);
    if (err != vvSocket::VV_OK)
    {
      return err;
    }

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Get a file name from the socket.
 @param fn  the file name.
*/
vvSocket::ErrorType vvSocketIO::getFileName(char*& fn)
{
  if(_socket)
  {
    uchar* buffer;
    vvSocket::ErrorType retval;

    buffer = new uchar[4];
    if ((retval =_socket->readData(buffer, 4)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    const size_t len = vvToolshed::read32(buffer);
    delete[] buffer;

    buffer = new uchar[len];
    if ((retval =_socket->readData(buffer, len)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }

    delete[] fn;
    fn = new char[len + 1];

    for (size_t i=0; i<len; ++i)
    {
      fn[i] = (char)buffer[i];
    }
    fn[len] = '\0';

    delete[] buffer;

    // TODO: check if this is really a file name... .

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write a file name to the socket.
 @param fn  the file name.
*/
vvSocket::ErrorType vvSocketIO::putFileName(const char* fn)
{
  if(_socket)
  {
    uchar* buffer;
    vvSocket::ErrorType retval;

    const size_t len = fn ? strlen(fn) : 0;
    buffer = new uchar[4 + len];
    vvToolshed::write32(&buffer[0], (uint32_t)len);

    for (size_t i=0; i<len; ++i)
    {
      buffer[4 + i] = (uchar)fn[i];
    }

    if ((retval =_socket->writeData(buffer, 4 + len)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }

    delete[] buffer;

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Gets arbitrary data of arbitrary size from the socket.
 @param data  pointer to the pointer where data shall be written. Memory is
 allocated which has to be deallocated outside this function.
 @param size  reference of an integer which includes the number of read bytes.
*/
vvSocket::ErrorType vvSocketIO::allocateAndGetData(uchar** data, int& size)
{
  if(_socket)
  {
    uchar buffer[4];
    vvSocket::ErrorType retval;

    *data = NULL; // make it safe to delete[] *data

    if ((retval =_socket->readData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Header received");
    size = (int)vvToolshed::read32(&buffer[0]);
    *data = new uchar[size];                        // delete buffer outside!!!
    if ((retval =_socket->readData(*data, size)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Data received");
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes data to a socket.
 @param data  pointer to the data which has to be written.
 @param size  number of bytes to write.
*/
vvSocket::ErrorType vvSocketIO::putData(uchar* data, int size)
{
  if(_socket)
  {
    uchar buffer[4];
    vvSocket::ErrorType retval;

    vvToolshed::write32(&buffer[0], (ulong)size);
    vvDebugMsg::msg(3, "Sending header ...");
    if ((retval =_socket->writeData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      return retval;
    }
    vvDebugMsg::msg(3, "Sending data ...");
    if ((retval =_socket->writeData(data, size)) != vvSocket::VV_OK)
    {
      return retval;
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Gets a fixed number of elements of a fixed type from the socket.
 @param data  pointer to where data shall be written.
 @param number  number of elements to read.
 @param type  data type to read. vvSocketIO::UCHAR for unsigned char,
 vvSocketIO::INT for integer and vvSocketIO::FLOAT for float.

*/
vvSocket::ErrorType vvSocketIO::getData(void* data, int number, DataType type)
{
  if(_socket)
  {
    vvSocket::ErrorType retval;
    int size;
    uchar* buffer;

    switch(type)
    {
      case VV_UCHAR:
      {
        size = number;
        if ((retval =_socket->readData((uchar*)data, size)) != vvSocket::VV_OK)
        {
          return retval;
        }
        vvDebugMsg::msg(3, "uchar received");
      }break;
      case VV_USHORT:
      {
        int tmp;
        size = number*2;
        buffer = new uchar[size];
        if ((retval =_socket->readData(buffer, size)) != vvSocket::VV_OK)
        {
          delete[] buffer;
          return retval;
        }
        for (int i=0; i<number; i++)
        {
          tmp = vvToolshed::read16(&buffer[i*2]);
          memcpy((uchar*)data+i*2, &tmp, 2);
        }
        vvDebugMsg::msg(3, "ushort received");
        delete[] buffer;
      }break;
      case VV_INT:
      {
        int tmp;
        size = number*4;
        buffer = new uchar[size];
        if ((retval =_socket->readData(buffer, size)) != vvSocket::VV_OK)
        {
          delete[] buffer;
          return retval;
        }
        for (int i=0; i<number; i++)
        {
          tmp = vvToolshed::read32(&buffer[i*4]);
          memcpy((uchar*)data+i*4, &tmp, 4);
        }
        vvDebugMsg::msg(3, "int received");
        delete[] buffer;
      }break;
      case VV_FLOAT:
      {
        float tmp;
        size = number*4;
        buffer = new uchar[size];
        if ((retval =_socket->readData(buffer, size)) != vvSocket::VV_OK)
        {
          delete[] buffer;
          return retval;
        }
        for (int i=0; i<number; i++)
        {
          tmp = vvToolshed::readFloat(&buffer[i*4]);
          memcpy((uchar*)data+i*4, &tmp, 4);
        }
        vvDebugMsg::msg(3, "float received");
        delete[] buffer;
      }break;
      default:
        vvDebugMsg::msg(0, "No supported data type");
        return vvSocket::VV_DATA_ERROR;
    }
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Write a number of fixed elements to a socket.
    @param data  pointer to the data to write.
    @param number  number of elements to write.
    @param type  data type to write. vvSocketIO::UCHAR for unsigned char,
    vvSocketIO::INT for integer and vvSocketIO::FLOAT for float.
*/
vvSocket::ErrorType vvSocketIO::putData(void* data, int number, DataType type)
{
  if(_socket)
  {
    vvSocket::ErrorType retval;
    int size;
    uchar* buffer;

    switch(type)
    {
      case VV_UCHAR:
      {
        size = number;
        buffer = (uchar*)data;
        vvDebugMsg::msg(3, "Sending uchar ...");
      }break;
      case (VV_USHORT):
      {
        int tmp;
        size = number*2;
        buffer = new uchar[size];

        for (int i=0; i<number; i++)
        {
          memcpy(&tmp, (uchar*)data+i*2 , 2);
          vvToolshed::write16(&buffer[i*2], (ushort)tmp);
        }
        vvDebugMsg::msg(3, "Sending ushort ...");
      }break;
      case (VV_INT):
      {
        int tmp;
        size = number*4;
        buffer = new uchar[size];

        for (int i=0; i<number; i++)
        {
          memcpy(&tmp, (uchar*)data+i*4 , 4);
          vvToolshed::write32(&buffer[i*4], (ulong)tmp);
        }
        vvDebugMsg::msg(3, "Sending integer ...");
      }break;
      case VV_FLOAT:
      {
        float tmp;
        size = number*4;
        buffer = new uchar[size];
        for (int i=0; i<number; i++)
        {
          memcpy(&tmp, (uchar*)data+i*4 , 4);
          vvToolshed::writeFloat(&buffer[i*4], (float)tmp);
        }
        vvDebugMsg::msg(3, "Sending float ...");
      }break;
      default:
        vvDebugMsg::msg(0, "No supported data type");
        return vvSocket::VV_DATA_ERROR;
    }
    if ((retval =_socket->writeData(buffer, size)) != vvSocket::VV_OK)
    {
      if (type != VV_UCHAR)
        delete[] buffer;
      return retval;
    }
    if (type != VV_UCHAR)
      delete[] buffer;
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Gets a Matrix from the socket.
    @param m  pointer to an object of vvMatrix.
*/
vvSocket::ErrorType vvSocketIO::getMatrix(vvMatrix* m)
{
  if(_socket)
  {
    uchar* buffer = NULL;
    int s;

    switch(allocateAndGetData(&buffer, s))
    {
      case vvSocket::VV_OK: break;
      case vvSocket::VV_DATA_ERROR: delete[] buffer; return vvSocket::VV_DATA_ERROR; break;
      default: delete[] buffer; return vvSocket::VV_DATA_ERROR;
    }
    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++)
        (*m)(i, j) = vvToolshed::readFloat(buffer+4*(4*i+j));
    delete[] buffer;
    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a boolean flag to the socket.
 @param val  the boolean flag.
*/
vvSocket::ErrorType vvSocketIO::putBool(const bool val)
{
  if(_socket)
  {
    uchar buffer[] = { (uchar)val };
    return _socket->writeData(&buffer[0], 1);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a boolean flag from the socket.
 @param val  the boolean flag.
*/
vvSocket::ErrorType vvSocketIO::getBool(bool& val)
{
  if(_socket)
  {
    uchar buffer[1];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 1)) != vvSocket::VV_OK)
    {
      return retval;
    }
    val = (buffer[0] != 0);

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes an int value to the socket.
 @param val  the int value.
*/
vvSocket::ErrorType vvSocketIO::putInt32(const int val)
{
  if(_socket)
  {
    uchar buffer[4];
    vvToolshed::write32(&buffer[0], val);
    return _socket->writeData(&buffer[0], 4);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads an int value from the socket.
 @param val  the int value.
*/
vvSocket::ErrorType vvSocketIO::getInt32(int& val)
{
  if(_socket)
  {
    uchar buffer[4];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val = vvToolshed::read32(&buffer[0]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a float value to the socket.
 @param val  the float value.
*/
vvSocket::ErrorType vvSocketIO::putFloat(const float val)
{
  if(_socket)
  {
    uchar buffer[4];
    vvToolshed::writeFloat(&buffer[0], val);
    return _socket->writeData(&buffer[0], 4);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a float value from the socket.
 @param val  the float value.
*/
vvSocket::ErrorType vvSocketIO::getFloat(float& val)
{
  if(_socket)
  {
    uchar buffer[4];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val = vvToolshed::readFloat(&buffer[0]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a vvVector3 to the socket.
 @param val  the vvVector3.
*/
vvSocket::ErrorType vvSocketIO::putVector3(const vvVector3& val)
{
  if(_socket)
  {
    uchar buffer[12];
    vvToolshed::writeFloat(&buffer[0], val[0]);
    vvToolshed::writeFloat(&buffer[4], val[1]);
    vvToolshed::writeFloat(&buffer[8], val[2]);
    return _socket->writeData(&buffer[0], 12);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a vvVector3 from the socket.
 @param val  the vvVector3.
*/
vvSocket::ErrorType vvSocketIO::getVector3(vvVector3& val)
{
  if(_socket)
  {
    uchar buffer[12];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 12)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val[0] = vvToolshed::readFloat(&buffer[0]);
    val[1] = vvToolshed::readFloat(&buffer[4]);
    val[2] = vvToolshed::readFloat(&buffer[8]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a vvVector4 to the socket.
 @param val  the vvVector4.
*/
vvSocket::ErrorType vvSocketIO::putVector4(const vvVector4& val)
{
  if(_socket)
  {
    uchar buffer[16];
    vvToolshed::writeFloat(&buffer[0], val[0]);
    vvToolshed::writeFloat(&buffer[4], val[1]);
    vvToolshed::writeFloat(&buffer[8], val[2]);
    vvToolshed::writeFloat(&buffer[12], val[3]);
    return _socket->writeData(&buffer[0], 16);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a vvVector4 from the socket.
 @param val  the vvVector4.
*/
vvSocket::ErrorType vvSocketIO::getVector4(vvVector4& val)
{
  if(_socket)
  {
    uchar buffer[16];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 16)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val[0] = vvToolshed::readFloat(&buffer[0]);
    val[1] = vvToolshed::readFloat(&buffer[4]);
    val[2] = vvToolshed::readFloat(&buffer[8]);
    val[3] = vvToolshed::readFloat(&buffer[12]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a vvAABBi to the socket.
 @param val  the vvAABBi.
*/
vvSocket::ErrorType vvSocketIO::putAABBi(const vvAABBi& val)
{
  if(_socket)
  {
    uchar buffer[24];
    const vvVector3i minval = val.getMin();
    const vvVector3i maxval = val.getMax();
    vvToolshed::write32(&buffer[0], minval[0]);
    vvToolshed::write32(&buffer[4], minval[1]);
    vvToolshed::write32(&buffer[8], minval[2]);
    vvToolshed::write32(&buffer[12], maxval[0]);
    vvToolshed::write32(&buffer[16], maxval[1]);
    vvToolshed::write32(&buffer[20], maxval[2]);
    return _socket->writeData(&buffer[0], 24);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a vvAABBi from the socket.
 @param val  the vvAABBi.
*/
vvSocket::ErrorType vvSocketIO::getAABBi(vvAABBi& val)
{
  if(_socket)
  {
    uchar buffer[24];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 24)) != vvSocket::VV_OK)
    {
      return retval;
    }

    vvVector3i minval;
    vvVector3i maxval;
    minval[0] = vvToolshed::read32(&buffer[0]);
    minval[1] = vvToolshed::read32(&buffer[4]);
    minval[2] = vvToolshed::read32(&buffer[8]);
    maxval[0] = vvToolshed::read32(&buffer[12]);
    maxval[1] = vvToolshed::read32(&buffer[16]);
    maxval[2] = vvToolshed::read32(&buffer[20]);
    val = vvAABBi(minval, maxval);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a vvGLTools::Viewport to the socket.
 @param val  the vvGLTools::Viewport.
*/
vvSocket::ErrorType vvSocketIO::putViewport(const vvGLTools::Viewport &val)
{
  if(_socket)
  {
    uchar buffer[16];
    vvToolshed::write32(&buffer[0], val[0]);
    vvToolshed::write32(&buffer[4], val[1]);
    vvToolshed::write32(&buffer[8], val[2]);
    vvToolshed::write32(&buffer[12], val[3]);
    return _socket->writeData(&buffer[0], 16);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a vvGLTools::Viewport from the socket.
 @param val  the vvGLTools::Viewport.
*/
vvSocket::ErrorType vvSocketIO::getViewport(vvGLTools::Viewport &val)
{
  if(_socket)
  {
    uchar buffer[16];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 16)) != vvSocket::VV_OK)
    {
      return retval;
    }

    val[0] = vvToolshed::read32(&buffer[0]);
    val[1] = vvToolshed::read32(&buffer[4]);
    val[2] = vvToolshed::read32(&buffer[8]);
    val[3] = vvToolshed::read32(&buffer[12]);
    return retval;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a comm reason to the socket.
 @param val  the comm reason.
*/
vvSocket::ErrorType vvSocketIO::putCommReason(const CommReason val)
{
  if(_socket)
  {
    uchar buffer[] = { (uchar)val };
    return _socket->writeData(&buffer[0], 4);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads a comm reason from the socket.
 @param val  the comm reason.
*/
vvSocket::ErrorType vvSocketIO::getCommReason(CommReason& val)
{
  if(_socket)
  {
    uchar buffer[4];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 4)) != vvSocket::VV_OK)
    {
      return retval;
    }
    val = (CommReason)buffer[0];

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::putWinDims(const int w, const int h)
{
  if(_socket)
  {
    uchar buffer[8];

    vvToolshed::write32(&buffer[0], w);
    vvToolshed::write32(&buffer[4], h);

    return _socket->writeData(&buffer[0], 8);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

vvSocket::ErrorType vvSocketIO::getWinDims(int& w, int& h)
{
  if(_socket)
  {
    uchar buffer[8];
    vvSocket::ErrorType retval;

    if ((retval =_socket->readData(&buffer[0], 8)) != vvSocket::VV_OK)
    {
      return  retval;
    }
    w = vvToolshed::read32(&buffer[0]);
    h = vvToolshed::read32(&buffer[4]);

    return vvSocket::VV_OK;
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes a Matrix to the socket.
 @param m  pointer to the matrix to write, has to be an object of vvMatrix.
*/
vvSocket::ErrorType vvSocketIO::putMatrix(const vvMatrix* m)
{
  if(_socket)
  {
    uchar buffer[64];

    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++)
        vvToolshed::writeFloat(&buffer[4*(4*i+j)], (*m)(i, j));
    return putData(buffer, 64);
  }
  else
  {
    return vvSocket::VV_SOCK_ERROR;
  }
}

//----------------------------------------------------------------------------
/** get assigned vvSocket
*/
vvSocket* vvSocketIO::getSocket() const
{
  return _socket;
}

//----------------------------------------------------------------------------
/** Allocate memory for a single brick.
*/
int vvSocketIO::sizeOfBrick() const
{
  // Assume integers and floats to be 4 byte long.
  return 3 * 4 // pos
       + 3 * 4 // min
       + 3 * 4 // max
       + 4 // minValue
       + 4 // maxValue
       + 1 // isVisible
       + 1 // atBorder
       + 1 // insideProbe
       + 1 // a padding for alignment, no data here
       + 4 // index
       + 3 * 4 // startOffset
       + 3 * 4 // texels
       + 1 * 4 // dist
       + 3 * 4 // texRange
       + 3 * 4; // texMin
}

// EOF
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
