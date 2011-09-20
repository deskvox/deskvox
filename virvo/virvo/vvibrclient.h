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

#ifndef _VV_IBRCLIENT_H_
#define _VV_IBRCLIENT_H_

#include "vvbsptree.h"
#include "vvexport.h"
#include "vvopengl.h"
#include "vvremoteclient.h"
#include "vvvoldesc.h"
#include "vvpthread.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvgltools.h"
#include "vvrayrend.h"

#include <vector>

class vvRenderer;
class vvSlaveVisitor;

class VIRVOEXPORT vvIbrClient : public vvRemoteClient
{
public:
  vvIbrClient(vvVolDesc *vd, vvRenderState renderState,
              const char* slaveNames, int slavePorts,
              const char* slaveFileNames,
              vvIbrImage::DepthPrecision dp = vvIbrImage::VV_USHORT);
  ~vvIbrClient();

  ErrorType render();                                     ///< render image with depth-values
  void exit();                                            ///< check out from servers

  void setDepthPrecision(vvIbrImage::DepthPrecision dp);  ///< set depth-value precision (1,2 or 4 bytes)
private:
  enum Corner
  {
    VV_TOP_LEFT = 0,
    VV_TOP_RIGHT,
    VV_BOTTOM_RIGHT,
    VV_BOTTOM_LEFT
  };

  //! thread-data
  struct ThreadArgs
  {
    vvIbrClient* renderMaster;
    std::vector<vvImage*> *images;
  };

  pthread_t*  _thread;                                    ///< list for threads of each server connection
  ThreadArgs* _threadData;                                ///< list for thread data

  pthread_mutex_t _slaveMutex;                            ///< mutex for thread synchronization
  bool   _newFrame;                                       ///< flag indicating a new ibr-frame waiting to be rendered
  bool   _haveFrame;                                      ///< flag indicating that at least one frame has been received
  GLuint _pointVBO;                                       ///< Vertex Buffer Object id for point-pixels
  GLuint _indexBO[4];                                     ///< Buffer Object ids for indices into points

  GLuint _rgbaTex;                                        ///< Texture names for RGBA image
  GLuint _depthTex;                                       ///< Texture names for depth image

  std::vector<GLuint> _indexArray[4];                     ///< four possible traversal directions for drawing the vertices

  vvMatrix _currentMv;                                    ///< Current modelview matrix
  vvMatrix _currentPr;                                    ///< Current projection matrix
  vvIbrImage* _ibrImg;                                    ///< Img retrieved from server
  vvRemoteClient::ErrorType requestIbrFrame();            ///< remember envoironment and send image-request to server
  void initIbrFrame();                                    ///< initialize pixel-points in object space

  vvIbrImage::DepthPrecision _depthPrecision;             ///< deph-value precision
  int _width, _height;                                    ///< dimensions of ibr image

  vvShaderFactory* _shaderFactory;
  vvShaderProgram* _shader;

  void initIndexArrays();                                 ///< initialize four index arrays for back to front traversal
  Corner getNearestCorner() const;                        ///< find the ibr-img corner with the shortest dist to the viewer
  void createImages();                                    ///< create an image to be read from, another to be written to
  void createThreads();                                   ///< creates threads for every socket connection
  void destroyThreads();                                  ///< quits threads
  static void* getImageFromSocket(void* threadargs);      ///< get image from socket connection and wait for next
};

#endif
