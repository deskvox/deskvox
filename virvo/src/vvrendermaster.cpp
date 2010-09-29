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

#include "vvbsptreevisitors.h"
#include "vvgltools.h"
#include "vvrendermaster.h"
#include "vvtexrend.h"

vvRenderMaster::vvRenderMaster(std::vector<char*>& slaveNames, std::vector<int>& slavePorts,
                               std::vector<char*>& slaveFileNames,
                               const char* fileName)
  : _slaveNames(slaveNames), _slavePorts(slavePorts),
  _slaveFileNames(slaveFileNames), _fileName(fileName)
{
  _visitor = new vvSlaveVisitor();
}

vvRenderMaster::~vvRenderMaster()
{
  // The visitor will delete the sockets either.
  delete _visitor;
}

vvRenderMaster::ErrorType vvRenderMaster::initSockets(const int defaultPort, vvSocket::SocketType st,
                                                      const bool redistributeVolData,
                                                      vvVolDesc*& vd)
{
  const bool loadVolumeFromFile = !redistributeVolData;
  for (int s=0; s<_slaveNames.size(); ++s)
  {
    if (_slavePorts[s] == -1)
    {
      _sockets.push_back(new vvSocketIO(defaultPort, _slaveNames[s], st));
    }
    else
    {
      _sockets.push_back(new vvSocketIO(_slavePorts[s], _slaveNames[s], st));
    }
    _sockets[s]->set_debuglevel(vvDebugMsg::getDebugLevel());
    _sockets[s]->no_nagle();

    if (_sockets[s]->init() == vvSocket::VV_OK)
    {
      _sockets[s]->putBool(loadVolumeFromFile);

      if (loadVolumeFromFile)
      {
        const bool allFileNamesAreEqual = (_slaveFileNames.size() == 0);
        if (allFileNamesAreEqual)
        {
          _sockets[s]->putFileName(_fileName);
        }
        else
        {
          if (_slaveFileNames.size() > s)
          {
            _sockets[s]->putFileName(_slaveFileNames[s]);
          }
          else
          {
            // Not enough file names specified, try this one.
            _sockets[s]->putFileName(_fileName);
          }
        }
      }
      else
      {
        switch (_sockets[s]->putVolume(vd))
        {
        case vvSocket::VV_OK:
          cerr << "Volume transferred successfully" << endl;
          break;
        case vvSocket::VV_ALLOC_ERROR:
          cerr << "Not enough memory" << endl;
          return VV_SOCKET_ERROR;
        default:
          cerr << "Cannot write volume to socket" << endl;
          return VV_SOCKET_ERROR;
        }
      }
    }
    else
    {
      cerr << "No connection to remote rendering server established at: " << _slaveNames[0] << endl;
      cerr << "Falling back to local rendering" << endl;
      return VV_SOCKET_ERROR;
    }
  }
  _visitor->setSockets(_sockets);
  return VV_OK;
}

vvRenderMaster::ErrorType vvRenderMaster::initBricks(vvTexRend* renderer)
{
  // This will build up the bsp tree of the master node.
  renderer->prepareDistributedRendering(_slaveNames.size());

  // Store a pointer to the bsp tree and set its visitor.
  _bspTree = renderer->getBspTree();
  _bspTree->setVisitor(_visitor);

  _renderer = renderer;

  // Distribute the bricks from the bsp tree
  for (int s=0; s<_sockets.size(); ++s)
  {
    switch (_sockets[s]->putBricks(renderer->getBrickListsToDistribute()[s]->at(0)))
    {
    case vvSocket::VV_OK:
      cerr << "Brick outlines transferred successfully" << endl;
      break;
    default:
      cerr << "Unable to transfer brick outlines" << endl;
      return VV_SOCKET_ERROR;
    }
  }
  return VV_OK;
}

void vvRenderMaster::render(const float bgColor[3]) const
{
  float matrixGL[16];

  vvMatrix pr;
  glGetFloatv(GL_PROJECTION_MATRIX, matrixGL);
  pr.set(matrixGL);

  vvMatrix mv;
  glGetFloatv(GL_MODELVIEW_MATRIX, matrixGL);
  mv.set(matrixGL);

  _renderer->calcProjectedScreenRects();

  _visitor->setProjectionMatrix(pr);
  _visitor->setModelviewMatrix(mv);

  glDrawBuffer(GL_BACK);
  glClearColor(bgColor[0], bgColor[1], bgColor[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Retrieve the eye position for bsp-tree traversal
  vvVector3 eye;
  _renderer->getEyePosition(&eye);
  vvMatrix invMV(&mv);
  invMV.invert();
  // This is a gl matrix ==> transpose.
  invMV.transpose();
  eye.multiply(&invMV);

  // Orthographic projection.
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  // Fix the proxy quad for the frame buffer texture.
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Setup compositing.
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

  _bspTree->traverse(eye);

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void vvRenderMaster::adjustQuality(const float quality)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_QUALITY) == vvSocket::VV_OK)
    {
      _sockets[s]->putFloat(quality);
    }
  }
}

void vvRenderMaster::resize(const int w, const int h)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_RESIZE) == vvSocket::VV_OK)
    {
      _sockets[s]->putWinDims(w, h);
    }
  }
}

void vvRenderMaster::setInterpolation(const bool interpolation)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_INTERPOLATION) == vvSocket::VV_OK)
    {
      _sockets[s]->putBool(interpolation);
    }
  }
}

void vvRenderMaster::setMipMode(const int mipMode)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_MIPMODE) == vvSocket::VV_OK)
    {
      _sockets[s]->putInt32(mipMode);
    }
  }
}

void vvRenderMaster::toggleBoundingBox()
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    _sockets[s]->putCommReason(vvSocketIO::VV_TOGGLE_BOUNDINGBOX);
  }
}
