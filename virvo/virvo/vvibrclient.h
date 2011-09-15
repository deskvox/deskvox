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
              std::vector<const char*>& slaveNames, std::vector<int>& slavePorts,
              std::vector<const char*>& slaveFileNames,
              vvImage2_5d::DepthPrecision dp = vvImage2_5d::VV_USHORT,
              vvImage2_5d::IbrDepthScale ds = vvImage2_5d::VV_FULL_DEPTH);
  ~vvIbrClient();

  ErrorType render();                                     ///< render image with depth-values
  void exit();                                            ///< check out from servers

  void setDepthPrecision(vvImage2_5d::DepthPrecision dp); ///< set depth-value precision (1,2 or 4 bytes)
  void setParameter(vvRenderer::ParameterType param, float newValue);
  void updateTransferFunction(vvTransFunc& tf);
private:
  //! thread-data
  struct ThreadArgs
  {
    int threadId;
    vvIbrClient* renderMaster;
    std::vector<vvImage*>* images;
  };

  pthread_t*  _threads;                   ///< list for threads of each server connection
  ThreadArgs* _threadData;                ///< list for thread data

  pthread_mutex_t _slaveMutex;            ///< mutex for thread synchronization
  bool   _slaveRdy;                       ///< flag to indicate that all servers are ready
  int    _slaveCnt;                       ///< counter for servers
  bool   _changes;                        ///< flag indicating changes that need a rendering-refresh
  bool   _newFrame;                       ///< flag indicating a new ibr-frame waiting to be rendered
  bool   _firstFrame;                     ///< flag indicating that the very first frame is drawn
  GLuint _pointVBO;                       ///< Vertex Buffer Object id for point-pixels
  GLuint _colorVBO;                       ///< Vertex Buffer Object id for pixel-colors

  GLuint _ibrTex[3];                      ///< Texture names for ibr-textures

  vvRect*             _isaRect[2];        ///< array for memorizing and flipping old and new screenrects
  vvGLTools::Viewport _vp[2];             ///< array for memorizing and flipping old and new viewport
  GLdouble            _modelMatrix[32];   ///< array for memorizing and flipping old and new modelview-matrix
  GLdouble            _projMatrix[32];    ///< array for memorizing and flipping old and new projection-matrix
  vvRemoteClient::ErrorType requestIbrFrame();  ///< remember envoironment and send image-request to server
  void initIbrFrame();                    ///< initialize pixel-points in object space

  vvImage2_5d::DepthPrecision _depthPrecision;        ///< deph-value precision
  vvImage2_5d::IbrDepthScale  _depthScale;
  float _ibrPlanes[4];

  vvShaderFactory* _shaderFactory;
  vvShaderProgram* _shader;

  void createThreads();                               ///< creates threads for every socket connection
  void destroyThreads();                              ///< quits threads
  static void* getImageFromSocket(void* threadargs);  ///< get image from socket connection and wait for next
};

#endif
