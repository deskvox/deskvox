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

#ifndef _VV_RENDERMASTER_H_
#define _VV_RENDERMASTER_H_

#include "vvbsptree.h"
#include "vvexport.h"
#include "vvopengl.h"
#include "vvremoteclient.h"
#include "vvvoldesc.h"
#include "vvpthread.h"

#include <vector>

class vvRenderer;
class vvSlaveVisitor;

class VIRVOEXPORT vvIsaClient : public vvRemoteClient
{
public:

  vvMatrix _mv;
  vvMatrix _pr;

  bool _gapStart;

  vvVector3 _eye;
  GLuint _pointVBO;
  GLuint _colorVBO;

  vvIsaClient(std::vector<const char*>& slaveNames, std::vector<int>& slavePorts,
                 std::vector<const char*>& slaveFileNames,
                 const char* fileName);
  ~vvIsaClient();

  ErrorType setRenderer(vvRenderer* renderer);
  ErrorType render();
  void exit();

  void resize(int w, int h);
  void setCurrentFrame(int index);
  void setMipMode(const int mipMode);
  void setObjectDirection(const vvVector3* od);
  void setViewingDirection(const vvVector3* vd);
  void setPosition(const vvVector3* p);
  void setROIEnable(bool roiEnabled);
  void setProbePosition(const vvVector3* pos);
  void setProbeSize(const vvVector3* roiSize);
  void toggleBoundingBox();
  void updateTransferFunction(vvTransFunc& tf);
  void setParameter(vvRenderer::ParameterType param, float newValue, const char* = NULL);
  void setISA(const bool isa);

private:
  vvBspTree* _bspTree;
  vvSlaveVisitor* _visitor;

  struct ThreadArgs
  {
    int threadId;
    vvIsaClient* renderMaster;
    std::vector<vvImage*>* images;
  };

  pthread_t* _threads;
  ThreadArgs* _threadData;
  pthread_barrier_t _barrier;

  void adjustQuality(float quality);
  void setInterpolation(bool interpolation);

  // Mutex to count socket-threads that are ready
  pthread_mutex_t _slaveMutex;
  bool _slaveRdy;
  int  _slaveCnt;

  bool    _imagespaceApprox;
  vvRect* _isaRect[2];
  void    initIsaFrame();

  void createThreads();
  void destroyThreads();
  static void* getImageFromSocket(void* threadargs);

};

#endif
