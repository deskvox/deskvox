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

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

using namespace std;

//----------------------------------------------------------------------------
/** Constructor for client.
 @param port  server port to connect to .
 @param servername  name of server to connect to.
 @param st  type of socket to create. vvSocket::VV_TCP for TCP, vvSocket::VV_UDP for UDP.
 @param clminport  minimum outgoing port.
 @param clmaxport  maximum outgoing port.
*/
vvSocketIO::vvSocketIO(int port, char* servername, vvSocket::SocketType st,int clminport, int clmaxport)
: vvSocket(port, servername, st, clminport, clmaxport)
{
}

//----------------------------------------------------------------------------
/** Constructor for server.
 @param port  port to listen.
 @param st  type of socket to create. vvSocket::VV_TCP for TCP,
 vvSocket::VV_UDP for UDP.
*/
vvSocketIO::vvSocketIO(int port, vvSocket::SocketType st) : vvSocket(port, st)
{
}

//----------------------------------------------------------------------------
/// Destructor
vvSocketIO::~vvSocketIO()
{
}

//----------------------------------------------------------------------------
/** Initializes a socket.
 */
vvSocket::ErrorType vvSocketIO::init()
{
  vvSocket::ErrorType retval;

  if ((retval = vvSocket::init()) != vvSocket::VV_OK)
  {
    return retval;
  }
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Sets the parameters for a socket. Has to be called before the init() call!!
 @param c_timer  timer for connection establishment.
 @param t_timer  timer for data transfer.
 @param sock_buff  size of socket buffers. For TCP automatic socket buffer
 size measurement if sock_buff=0.
 @param level  debuglevel. 0 for no messages, 3 for maximal debugging.
*/
void vvSocketIO::set_sock_param(float c_timer, float t_timer, int sock_buff, int level)
{
  vvSocket::set_timer(c_timer, t_timer);
  vvSocket::set_sock_buffsize(sock_buff);
  vvSocket::set_debuglevel(level);
}

//----------------------------------------------------------------------------
/** Checks if there is data in the socket receive buffer.
 @return  true for data in the socket receive buffer, false for not.
*/
bool vvSocketIO::sock_action()
{
  if (vvSocket::is_data_waiting() > 0)
    return true;
  else
    return false;
}

//----------------------------------------------------------------------------
/** Get volume data from socket.
  @param vd  empty volume description which is to be filled with the volume data
*/
vvSocket::ErrorType vvSocketIO::getVolume(vvVolDesc* vd)
{

  int size;
  uchar* buffer;
  vvSocket::ErrorType retval;

  buffer = new uchar[3];
                                                  // get extension
  if ((retval = vvSocket::read_data(buffer, 3)) != vvSocket::VV_OK)
  {
    delete[] buffer;
    return retval;
  }
  if ((char)buffer[0] == 'r' || (char)buffer[0] == 'R')
  {
    if (vvDebugMsg::isActive(3))
      cerr <<"Data type: rvf"<< endl;
    delete[] buffer;
    buffer = new uchar[6];
                                                  // get header
    if ((retval = vvSocket::read_data(buffer, 6)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    if (vvDebugMsg::isActive(3))
      cerr<<"Header received"<< endl;
    vd->vox[0] = vvToolshed::read16(&buffer[0]);
    vd->vox[1] = vvToolshed::read16(&buffer[2]);
    vd->vox[2] = vvToolshed::read16(&buffer[4]);
    vd->bpc = 1;
    vd->chan = 1;
    vd->frames = 1;
    vd->setFilename("default.rvf");
    delete[] buffer;

  }
  else
  {
    if (vvDebugMsg::isActive(3))
      cerr <<"Data type: xvf"<< endl;
    delete[] buffer;
    size = vd->serializeAttributes();
    cerr<<size<<endl;
    buffer = new uchar[size];
    if ((retval = vvSocket::read_data(buffer, size)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    if (vvDebugMsg::isActive(3))
      cerr<<"Header received"<< endl;
    vd->deserializeAttributes(buffer);
    delete[] buffer;

  }
  size = vd->getFrameBytes();
  for(int k =0; k< vd->frames; k++)
  {
    buffer = new uchar[size];
    if (!buffer)
      return vvSocket::VV_ALLOC_ERROR;
    if ((retval = vvSocket::read_data(buffer, size)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    vd->addFrame(buffer, vvVolDesc::ARRAY_DELETE);
  }
  if (vvDebugMsg::isActive(3))
    cerr<<"Data received"<<endl;
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Write volume data to socket.
  @param vd  volume description of volume to be send.
*/
vvSocket::ErrorType vvSocketIO::putVolume(vvVolDesc* vd)
{
  const char* filename;
  char ext[16];
  int size, frames;
  uchar* buffer;
  vvSocket::ErrorType retval;

  filename = vd->getFilename();
  vvToolshed::extractExtension(ext, filename);

  buffer = new uchar[3];
  for (int i=0; i<3; i++)
    buffer[i] = (uchar)ext[i];
  if (vvDebugMsg::isActive(3))
    cerr<<"Sending extension ..."<<endl;
  if ((retval = vvSocket::write_data(buffer, 3)) != vvSocket::VV_OK)
  {
    delete[] buffer;
    return retval;
  }

  delete[] buffer;

  if (ext[0] == 'r' || ext[0] == 'R')
  {
    buffer = new uchar[6];
    vvToolshed::write16(&buffer[0], (short)vd->vox[0]);
    vvToolshed::write16(&buffer[2], (short)vd->vox[1]);
    vvToolshed::write16(&buffer[4], (short)vd->vox[2]);
    if (vvDebugMsg::isActive(3))
      cerr <<"Sending header ..."<< endl;
    if ((retval = vvSocket::write_data(buffer, 6)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    delete[] buffer;
    frames=1;
  }
  else
  {
    size = vd->serializeAttributes();
    buffer = new uchar[size];
    vd->serializeAttributes(buffer);
    if (vvDebugMsg::isActive(3))
      cerr <<"Sending header ..."<< endl;
    if ((retval = vvSocket::write_data(buffer, size)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      return retval;
    }
    delete[] buffer;
    frames = vd->frames;
  }

  size = vd->getFrameBytes();
  if (vvDebugMsg::isActive(3))
    cerr <<"Sending data ..."<< endl;
  for(int k=0; k < frames; k++)
  {
    buffer = vd->getRaw(k);
    if ((retval = vvSocket::write_data(buffer, size)) != vvSocket::VV_OK)
    {
      return retval;
    }
  }
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Get a single brick from the socket.
  @param brick  pointer to a vvBrick.
*/
vvSocket::ErrorType vvSocketIO::getBrick(vvBrick* brick)
{
  uchar* buffer;
  vvSocket::ErrorType retval;

  const int sob = sizeOfBrick();
  buffer = new uchar[sob];

  if ((retval = vvSocket::read_data(buffer, sob)) != vvSocket::VV_OK)
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

  brick->visible = (bool)buffer[44];
  brick->insideProbe = (bool)buffer[46];
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

  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Write a single brick to the socket.
  @param brick  pointer to a vvBrick.
*/
vvSocket::ErrorType vvSocketIO::putBrick(vvBrick* brick)
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

  if ((retval = vvSocket::write_data(buffer, sob)) != vvSocket::VV_OK)
  {
    delete[] buffer;
    return retval;
  }

  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Get brick list from the socket. Bricks contain no volume data.
  @param bricks  std::vector with pointers to bricks.
*/
vvSocket::ErrorType vvSocketIO::getBricks(std::vector<vvBrick*>& bricks)
{
  uchar* buffer;
  vvSocket::ErrorType retval;

  buffer = new uchar[4];

  if ((retval = vvSocket::read_data(buffer, 4)) != vvSocket::VV_OK)
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

//----------------------------------------------------------------------------
/** Write brick list to the socket. Bricks contain no volume data.
  @param bricks  std::vector with pointers to bricks.
*/
vvSocket::ErrorType vvSocketIO::putBricks(std::vector<vvBrick*>& bricks)
{
  uchar* buffer;
  vvSocket::ErrorType retval;

  const int numBricks = (const int)bricks.size();

  buffer = new uchar[4];
  vvToolshed::write32(&buffer[0], numBricks);

  if (vvDebugMsg::isActive(3))
    cerr<<"Sending num bricks ..."<<endl;
  if ((retval = vvSocket::write_data(&buffer[0], 4)) != vvSocket::VV_OK)
  {
    delete[] buffer;
    return retval;
  }
  delete[] buffer;

  for(std::vector<vvBrick*>::iterator it = bricks.begin(); it != bricks.end(); ++it)
  {
    vvBrick* brick = (*it);
    if ((retval = putBrick(brick)) != vvSocket::VV_OK)
    {
      return retval;
    }
  }
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Get an image from the socket.
 @param im  pointer to a vvImage object.
*/
vvSocket::ErrorType vvSocketIO::getImage(vvImage* im)
{
  const int BUFSIZE = 17;
  uchar buffer[BUFSIZE];
  vvSocket::ErrorType retval;
  short w, h, ct;
  int imagesize;
  int videosize;
  int keyframe;

  if ((retval = vvSocket::read_data(&buffer[0], BUFSIZE)) != vvSocket::VV_OK)
  {
    return retval;
  }
  if (vvDebugMsg::isActive(3))
    cerr<<"Header received"<< endl;
  w = vvToolshed::read16(&buffer[2]);
  h = vvToolshed::read16(&buffer[0]);
  ct = (short)vvToolshed::read8(&buffer[4]);
  im->setCodeType(ct);
  if (h != im->getHeight() || w  != im->getWidth() || ct != im->getCodeType() )
  {
    im->setCodeType(ct);
    im->setHeight(h);
    im->setWidth(w);
    if(im->alloc_mem())
      return vvSocket::VV_ALLOC_ERROR;
  }
  imagesize = (int)vvToolshed::read32(&buffer[5]);
  keyframe = (int)vvToolshed::read32(&buffer[9]);
  videosize = (int)vvToolshed::read32(&buffer[13]);
  im->setSize(imagesize);
  im->setKeyframe(keyframe);
  im->setVideoSize(videosize);
  if (vvDebugMsg::isActive(3))
    fprintf(stderr, "imgsize=%d, keyframe=%d, videosize=%d\n", imagesize, keyframe, videosize);
  if ((retval = vvSocket::read_data(im->getCodedImage(), imagesize)) != vvSocket::VV_OK)
  {
    return retval;
  }
  if (vvDebugMsg::isActive(3))
    cerr<<"Image data received"<<endl;
  if (ct == 3)
  {
    if ((retval = vvSocket::read_data(im->getVideoCodedImage(), videosize)) != vvSocket::VV_OK)
    {
      return retval;
    }
    if (vvDebugMsg::isActive(3))
      cerr<<"Video Image data received"<<endl;
  }
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Write an image to the socket.
 @param im  pointer to an vvImage object.
*/
vvSocket::ErrorType vvSocketIO::putImage(vvImage* im)
{
  const int BUFSIZE = 17;
  uchar buffer[BUFSIZE];
  vvSocket::ErrorType retval;
  int imagesize;
  int videosize;
  int keyframe;
  int ct;

  imagesize = im->getSize();
  videosize = im->getVideoSize();
  keyframe = im->getKeyframe();
  ct = im->getCodeType();
  vvToolshed::write16(&buffer[0], im->getHeight());
  vvToolshed::write16(&buffer[2], im->getWidth());
  vvToolshed::write8(&buffer[4], (uchar)ct);
  vvToolshed::write32(&buffer[5], (ulong)imagesize);
  vvToolshed::write32(&buffer[9], (ulong)keyframe);
  vvToolshed::write32(&buffer[13], (ulong)videosize);
  if (vvDebugMsg::isActive(3))
    cerr <<"Sending header ..."<< endl;
  if ((retval = vvSocket::write_data(&buffer[0], BUFSIZE)) != vvSocket::VV_OK)
  {
    return retval;
  }
  if (vvDebugMsg::isActive(3))
    cerr <<"Sending image data ..."<< endl;
  if ((retval = vvSocket::write_data(im->getImagePtr(), imagesize)) != vvSocket::VV_OK)
  {
    return retval;
  }
  if (ct == 3)
  {
    if (vvDebugMsg::isActive(3))
      cerr <<"Sending video image data ..."<< endl;
    if ((retval = vvSocket::write_data(im->getVideoCodedImage(), videosize)) != vvSocket::VV_OK)
    {
      return retval;
    }
  }
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Get a file name from the socket.
 @param fn  the file name.
*/
vvSocket::ErrorType vvSocketIO::getFileName(char*& fn)
{
  uchar* buffer;
  vvSocket::ErrorType retval;

  buffer = new uchar[4];
  if ((retval = vvSocket::read_data(buffer, 4)) != vvSocket::VV_OK)
  {
    delete[] buffer;
    return retval;
  }
  const size_t len = vvToolshed::read32(buffer);
  delete[] buffer;

  buffer = new uchar[len];
  if ((retval = vvSocket::read_data(buffer, len)) != vvSocket::VV_OK)
  {
    delete[] buffer;
    return retval;
  }

  fn = new char[len + 1];

  for (size_t i=0; i<len; ++i)
  {
    fn[i] = (char)buffer[i];
  }
  fn[len] = '\0';

  // TODO: check if this is really a file name... .

  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Write a file name to the socket.
 @param fn  the file name.
*/
vvSocket::ErrorType vvSocketIO::putFileName(const char* fn)
{
  uchar* buffer;
  vvSocket::ErrorType retval;

  const size_t len = strlen(fn);
  buffer = new uchar[4 + len];
  vvToolshed::write32(&buffer[0], len);

  for (size_t i=0; i<len; ++i)
  {
    buffer[4 + i] = (uchar)fn[i];
  }

  if ((retval = vvSocket::write_data(buffer, 4 + len)) != vvSocket::VV_OK)
  {
    delete[] buffer;
    return retval;
  }
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Gets arbitrary data of arbitrary size from the socket.
 @param data  pointer to the pointer where data shall be written. Memory is
 allocated which has to be deallocated outside this function.
 @param size  reference of an integer which includes the number of read bytes.
*/
vvSocket::ErrorType vvSocketIO::getData(uchar** data, int& size)
{
  uchar buffer[4];
  vvSocket::ErrorType retval;

  if ((retval = vvSocket::read_data(&buffer[0], 4)) != vvSocket::VV_OK)
  {
    return retval;
  }
  if (vvDebugMsg::isActive(3))
    cerr<<"Header received"<< endl;
  size = (int)vvToolshed::read32(&buffer[0]);
  *data = new uchar[size];                        // delete buffer outside!!!
  if ((retval = vvSocket::read_data(*data, size)) != vvSocket::VV_OK)
  {
    return retval;
  }
  if (vvDebugMsg::isActive(3))
    cerr<<"Data received"<<endl;
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Writes data to a socket.
 @param data  pointer to the data which has to be written.
 @param size  number of bytes to write.
*/
vvSocket::ErrorType vvSocketIO::putData(uchar* data, int size)
{
  uchar buffer[4];
  vvSocket::ErrorType retval;

  vvToolshed::write32(&buffer[0], (ulong)size);
  if (vvDebugMsg::isActive(3))
    cerr <<"Sending header ..."<< endl;
  if ((retval = vvSocket::write_data(&buffer[0], 4)) != vvSocket::VV_OK)
  {
    return retval;
  }
  if (vvDebugMsg::isActive(3))
    cerr <<"Sending data ..."<< endl;
  if ((retval = vvSocket::write_data(data, size)) != vvSocket::VV_OK)
  {
    return retval;
  }
  return vvSocket::VV_OK;
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
  vvSocket::ErrorType retval;
  int size;
  uchar* buffer;

  switch(type)
  {
    case VV_UCHAR:
    {
      size = number;
      if ((retval = vvSocket::read_data((uchar*)data, size)) != vvSocket::VV_OK)
      {
        return retval;
      }
      if (vvDebugMsg::isActive(3))
        cerr<<"uchar received"<<endl;
    }break;
    case VV_INT:
    {
      int tmp;
      size = number*4;
      buffer = new uchar[size];
      if ((retval = vvSocket::read_data(buffer, size)) != vvSocket::VV_OK)
      {
        delete[] buffer;
        return retval;
      }
      for (int i=0; i<number; i++)
      {
        tmp = vvToolshed::read32(&buffer[i*4]);
        memcpy((uchar*)data+i*4, &tmp, 4);
      }
      if (vvDebugMsg::isActive(3))
        cerr<<"int received"<<endl;
      delete[] buffer;
    }break;
    case VV_FLOAT:
    {
      float tmp;
      size = number*4;
      buffer = new uchar[size];
      if ((retval = vvSocket::read_data(buffer, size)) != vvSocket::VV_OK)
      {
        delete[] buffer;
        return retval;
      }
      for (int i=0; i<number; i++)
      {
        tmp = vvToolshed::readFloat(&buffer[i*4]);
        memcpy((uchar*)data+i*4, &tmp, 4);
      }
      if (vvDebugMsg::isActive(3))
        cerr<<"float received"<<endl;
      delete[] buffer;
    }break;
    default:
      cerr<<"No supported data type"<<endl;
      return vvSocket::VV_DATA_ERROR;
  }
  return vvSocket::VV_OK;
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
  vvSocket::ErrorType retval;
  int size;
  uchar* buffer;

  switch(type)
  {
    case VV_UCHAR:
    {
      size = number;
      buffer = (uchar*)data;
      if (vvDebugMsg::isActive(3))
        cerr <<"Sending uchar ..."<< endl;
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
      if (vvDebugMsg::isActive(3))
        cerr <<"Sending integer ..."<< endl;
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
      if (vvDebugMsg::isActive(3))
        cerr <<"Sending float ..."<< endl;

    }break;
    default:
      cerr<<"No supported data type"<<endl;
      return vvSocket::VV_DATA_ERROR;
  }
  if ((retval = vvSocket::write_data(buffer, size)) != vvSocket::VV_OK)
  {
    if (type != VV_UCHAR)
      delete[] buffer;
    return retval;
  }
  if (type != VV_UCHAR)
    delete[] buffer;
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Gets a Matrix from the socket.
    @param m  pointer to an object of vvMatrix.
*/
vvSocket::ErrorType vvSocketIO::getMatrix(vvMatrix* m)
{
  uchar* buffer;
  int s;

  switch(getData(&buffer, s))
  {
    case vvSocket::VV_OK: break;
    case vvSocket::VV_DATA_ERROR: delete[] buffer; return vvSocket::VV_DATA_ERROR; break;
    case vvSocket::VV_TIMEOUT_ERROR: delete[] buffer; return vvSocket::VV_TIMEOUT_ERROR;break;
    default: delete[] buffer; return vvSocket::VV_DATA_ERROR;
  }
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      m->e[i][j] = vvToolshed::readFloat(buffer+4*(4*i+j));
  delete[] buffer;
  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Writes a boolean flag to the socket.
 @param val  the boolean flag.
*/
vvSocket::ErrorType vvSocketIO::putBool(const bool val)
{
  uchar buffer[] = { (uchar)val };
  return vvSocket::write_data(&buffer[0], 1);
}

//----------------------------------------------------------------------------
/** Reads a boolean flag from the socket.
 @param val  the boolean flag.
*/
vvSocket::ErrorType vvSocketIO::getBool(bool& val)
{
  uchar buffer[1];
  vvSocket::ErrorType retval;

  if ((retval = vvSocket::read_data(&buffer[0], 1)) != vvSocket::VV_OK)
  {
    return retval;
  }
  val = (bool)buffer[0];

  return vvSocket::VV_OK;
}

//----------------------------------------------------------------------------
/** Writes a Matrix to the socket.
 @param m  pointer to the matrix to write, has to be an object of vvMatrix.
*/
vvSocket::ErrorType vvSocketIO::putMatrix(vvMatrix* m)
{
  uchar buffer[64];

  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      vvToolshed::writeFloat(&buffer[4*(4*i+j)], m->e[i][j]);
  return putData(buffer, 64);
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
