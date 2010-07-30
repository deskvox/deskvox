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

#include "vvgltools.h"
#include "vvrendermaster.h"
#include "vvtexrend.h"

vvRenderMaster::vvRenderMaster(std::vector<char*>& slaveNames, std::vector<char*>& slaveFileNames,
                               const char* fileName)
  : _slaveNames(slaveNames), _slaveFileNames(slaveFileNames), _fileName(fileName)
{
  glGenTextures(1, &_textureId);
}

vvRenderMaster::~vvRenderMaster()
{

}

vvRenderMaster::ErrorType vvRenderMaster::initSockets(const int port, vvSocket::SocketType st,
                                                      const bool redistributeVolData,
                                                      vvVolDesc*& vd)
{
  const bool loadVolumeFromFile = !redistributeVolData;
  for (int s=0; s<_slaveNames.size(); ++s)
  {
    _sockets.push_back(new vvSocketIO(port, _slaveNames[s], st));
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
  return VV_OK;
}

vvRenderMaster::ErrorType vvRenderMaster::initBricks(vvTexRend* renderer)
{
  // This will build up the bsp tree of the master node.
  renderer->prepareDistributedRendering(_slaveNames.size());

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

void vvRenderMaster::render(const float bgColor[3])
{
  float matrixGL[16];

  vvMatrix pr;
  glGetFloatv(GL_PROJECTION_MATRIX, matrixGL);
  pr.set(matrixGL);

  vvMatrix mv;
  glGetFloatv(GL_MODELVIEW_MATRIX, matrixGL);
  mv.set(matrixGL);

  const vvGLTools::Viewport viewport = vvGLTools::getViewport();

  for (int s=0; s<_sockets.size(); ++s)
  {
     _sockets[s]->putMatrix(&pr);
     _sockets[s]->putMatrix(&mv);

     vvImage img = vvImage(viewport[3], viewport[2], new uchar[viewport[3] * viewport[2] * 4]);
     _sockets[s]->getImage(&img);

     glDrawBuffer(GL_BACK);
     glClearColor(bgColor[0], bgColor[1], bgColor[2], 1.0f);
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

     // Orthographic projection.
     glMatrixMode(GL_PROJECTION);
     glPushMatrix();
     glLoadIdentity();

     // Fix the proxy quad for the frame buffer texture.
     glMatrixMode(GL_MODELVIEW);
     glPushMatrix();
     glLoadIdentity();

     glActiveTextureARB(GL_TEXTURE0_ARB);
     glEnable(GL_TEXTURE_2D);
     glBindTexture(GL_TEXTURE_2D, _textureId);
     glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
     glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
     glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.getWidth(), img.getHeight(),
                  0, GL_RGBA, GL_UNSIGNED_BYTE, img.getCodedImage());
     vvGLTools::drawViewAlignedQuad();

     glMatrixMode(GL_PROJECTION);
     glPopMatrix();

     glMatrixMode(GL_MODELVIEW);
     glPopMatrix();
  }
}
