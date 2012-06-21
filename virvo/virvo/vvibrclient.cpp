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

#include "vvglew.h"
#include "vvibr.h"
#include "vvibrclient.h"
#include "vvibrimage.h"
#include "vvgltools.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvtoolshed.h"
#include "vvsocketio.h"
#include "vvdebugmsg.h"
#include "vvvoldesc.h"
#include "vvpthread.h"

using std::cerr;
using std::endl;

struct vvIbrClient::Thread
{
  pthread_t thread;               ///< list for threads of each server connection
  pthread_barrier_t startBarrier; ///< signal when network thread has started
  pthread_mutex_t signalMutex;    ///< mutex for thread synchronization
  pthread_mutex_t imageMutex;     ///< mutex for access to _image
  pthread_cond_t imageCond;       ///< condition variable for access to _image
  pthread_cond_t readyCond;       ///< signal when image has been received

  Thread(int numThreads)
  {
    pthread_barrier_init(&startBarrier, NULL, numThreads);

    pthread_mutex_init(&imageMutex, NULL);
    pthread_mutex_init(&signalMutex, NULL);

    pthread_cond_init(&imageCond, NULL);
    pthread_cond_init(&readyCond, NULL);
  }

  ~Thread()
  {
    pthread_mutex_destroy(&imageMutex);
    pthread_mutex_destroy(&signalMutex);
    pthread_cond_destroy(&imageCond);
    pthread_barrier_destroy(&startBarrier);
    pthread_cond_destroy(&readyCond);
  }
};

vvIbrClient::vvIbrClient(vvVolDesc *vd, vvRenderState renderState,
                         vvTcpSocket* socket, const std::string& filename)
  : vvRemoteClient(vd, renderState, vvRenderer::REMOTE_IBR, socket, filename)
  , _thread(NULL)
  , _newFrame(true)
  , _haveFrame(false)
  , _synchronous(false)
  , _image(NULL)
  , _shader(NULL)
{
  vvDebugMsg::msg(1, "vvIbrClient::vvIbrClient()");

  rendererType = REMOTE_IBR;

  glewInit();

  if(glGenBuffers)
  {
    glGenBuffers(1, &_pointVBO);
  }
  else
  {
    vvDebugMsg::msg(0, "vvIbrClient::vvIbrClient: no support for buffer objects");
  }


  glGenTextures(1, &_rgbaTex);
  glGenTextures(1, &_depthTex);

  _haveFrame = false; // no rendered frame available
  _newFrame = true; // request a new frame
  _image = new vvIbrImage;

  _shader = vvShaderFactory().createProgram("ibr", "", "ibr");
  if(!_shader)
    vvDebugMsg::msg(0, "vvIbrClient::vvIbrClient: could not find ibr shader");

  createThreads();
}

vvIbrClient::~vvIbrClient()
{
  vvDebugMsg::msg(1, "vvIbrClient::~vvIbrClient()");

  destroyThreads();

  if(glDeleteBuffers)
    glDeleteBuffers(1, &_pointVBO);
  glDeleteTextures(1, &_rgbaTex);
  glDeleteTextures(1, &_depthTex);
  delete _shader;
  delete _image;
}

void vvIbrClient::setParameter(vvRenderer::ParameterType param, const vvParam& newValue)
{
  switch(param)
  {
  case VV_IBR_SYNC:
    _synchronous = newValue;
    break;
  default:
    // intentionally do nothing
    break;
  }
  vvRemoteClient::setParameter(param, newValue);
}


vvRemoteClient::ErrorType vvIbrClient::render()
{
  vvDebugMsg::msg(1, "vvIbrClient::render()");

  pthread_mutex_lock(&_thread->signalMutex);
  bool haveFrame = _haveFrame;
  bool newFrame = _newFrame;
  pthread_mutex_unlock(&_thread->signalMutex);

  // Draw boundary lines
  if (_boundaries || !haveFrame || !glGenBuffers)
  {
    const vvVector3 size(vd->getSize()); // volume size [world coordinates]
    drawBoundingBox(size, vd->pos, _boundColor);
  }

  if (_shader == NULL)
  {
    return vvRemoteClient::VV_SHADER_ERROR;
  }

  vvMatrix currentMatrix = _currentPr * _currentMv;

  if(newFrame && haveFrame)
  {
    pthread_mutex_lock(&_thread->imageMutex);
    initIbrFrame();
    pthread_mutex_unlock(&_thread->imageMutex);
  }

  float drMin = 0.0f;
  float drMax = 0.0f;
  vvAABB aabb = vvAABB(vvVector3(), vvVector3());
  vd->getBoundingBox(aabb);
  vvIbr::calcDepthRange(_currentPr, _currentMv, aabb, drMin, drMax);
  const vvGLTools::Viewport vp = vvGLTools::getViewport();
  vvMatrix currentImgMatrix = vvIbr::calcImgMatrix(_currentPr, _currentMv, vp, drMin, drMax);
  bool matrixChanged = (!currentImgMatrix.equal(_imgMatrix));

  if (newFrame) // no frame pending
  {
    _changes |= matrixChanged;

    if(_changes)
    {
      pthread_mutex_lock(&_thread->signalMutex);
      vvRemoteClient::ErrorType err = requestFrame();
      _newFrame = false;
      pthread_cond_signal(&_thread->imageCond);
      pthread_mutex_unlock(&_thread->signalMutex);
      _changes = false;
      if(err != vvRemoteClient::VV_OK)
        std::cerr << "vvibrClient::requestFrame() - error() " << err << std::endl;
      else if(_synchronous)
      {
        pthread_mutex_lock(&_thread->signalMutex);
        pthread_cond_wait(&_thread->readyCond, &_thread->signalMutex);
        haveFrame = _haveFrame;
        newFrame = _newFrame;
        matrixChanged = false;
        pthread_mutex_unlock(&_thread->signalMutex);
        if(newFrame && haveFrame)
        {
          pthread_mutex_lock(&_thread->imageMutex);
          initIbrFrame();
          pthread_mutex_unlock(&_thread->imageMutex);
        }
      }
    }
  }

  if(!haveFrame)
  {
    // no frame was yet received
    return VV_OK;
  }

  if(!glGenBuffers)
  {
    // rendering requires vertex buffer objects
    return VV_GL_ERROR;
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // prepare VBOs
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glVertexPointer(3, GL_FLOAT, 0, NULL);
  glEnableClientState(GL_VERTEX_ARRAY);

  vvMatrix reprojectionMatrix;
  if (!matrixChanged)
  {
    reprojectionMatrix.identity();
  }
  else
  {
    vvMatrix invOld = _imgPr * _imgMv;
    invOld.invert();
    reprojectionMatrix = currentMatrix * invOld;
  }
  vvMatrix invMv = _currentMv;
  invMv.invert();
  vvVector4 viewerObj(0.f, 0.f, 0.f, 1.f);
  viewerObj.multiply(invMv);
  viewerObj.multiply(_imgMv);
  bool closer = viewerObj[2] > 0.f; // inverse render order if viewer has moved closer

  // project current viewer onto original image along its normal
  viewerObj.multiply(_imgPr);
  float splitX = (viewerObj[0]/viewerObj[3]+1.f)*_imgVp[2]*0.5f;
  float splitY = (viewerObj[1]/viewerObj[3]+1.f)*_imgVp[3]*0.5f;
  splitX = ts_clamp(splitX, 0.f, float(_imgVp[2]-1));
  splitY = ts_clamp(splitY, 0.f, float(_imgVp[3]-1));

  GLboolean depthMask = GL_TRUE;
  glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
  glDepthMask(GL_FALSE);
  GLboolean pointSmooth = GL_FALSE;
  glGetBooleanv(GL_POINT_SMOOTH, &pointSmooth);
  //glEnable(GL_POINT_SMOOTH);

  glEnable(GL_POINT_SPRITE);
  glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

  _shader->enable();

  _shader->setParameter1f("vpWidth", static_cast<float>(_viewportWidth));
  _shader->setParameter1f("vpHeight", static_cast<float>(_viewportHeight));
  _shader->setParameter1f("imageWidth", static_cast<float>(_imgVp[2]));
  _shader->setParameter1f("imageHeight", static_cast<float>(_imgVp[3]));
  _shader->setParameterTex2D("rgbaTex", _rgbaTex);
  _shader->setParameterTex2D("depthTex", _depthTex);
  _shader->setParameter1f("splitX", splitX);
  _shader->setParameter1f("splitY", splitY);
  _shader->setParameter1f("depthMin", _imgDepthRange[0]);
  _shader->setParameter1f("depthRange", _imgDepthRange[1]-_imgDepthRange[0]);
  _shader->setParameter1i("closer", closer);
  _shader->setParameterMatrix4f("reprojectionMatrix" , reprojectionMatrix);

  //// begin ellipsoid test code - temporary

  // hardwired parameters for now
  float v_i[16] = {
    0.70710678f, 0.70710678f, 0.0f, 0.0f,
   -0.70710678f, 0.70710678f, 0.0f, 0.0f,
           0.0f,        0.0f, 1.0f, 0.0f,
           0.0f,        0.0f, 0.0f, 1.0f
  };
  vvMatrix V_i = vvMatrix(v_i);
  V_i.invert();

  _shader->setParameter1f("si", 10.0);
  _shader->setParameter1f("sj", 10.0);
  _shader->setParameter1f("sk", 10.0);
  _shader->setParameterMatrix4f("V_i" , V_i);
  //// end ellipsoid test code - temporary

  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  glDrawArrays(GL_POINTS, 0, _imgVp[2]*_imgVp[3]);

  _shader->disable();

  if(depthMask)
    glDepthMask(GL_TRUE);
  if(!pointSmooth)
    glDisable(GL_POINT_SMOOTH);

  glDisableClientState(GL_VERTEX_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  return VV_OK;
}

void vvIbrClient::initIbrFrame()
{
  vvDebugMsg::msg(1, "vvIbrClient::initIbrFrame()");

  const int h = _image->getHeight();
  const int w = _image->getWidth();
  _imgMatrix = _image->getReprojectionMatrix();
  _imgPr = _image->getProjectionMatrix();
  _imgMv = _image->getModelViewMatrix();
  _image->getDepthRange(&_imgDepthRange[0], &_imgDepthRange[1]);

  // get pixel and depth-data
  glBindTexture(GL_TEXTURE_2D, _rgbaTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, _image->getImagePtr());

  glBindTexture(GL_TEXTURE_2D, _depthTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  uchar* depth = _image->getPixelDepth();
  switch(_image->getDepthPrecision())
  {
    case 8:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, depth);
      break;
    case 16:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT, depth);
      break;
    case 32:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, w, h, 0, GL_LUMINANCE, GL_FLOAT, depth);
      break;
  }

  if(_imgVp[2] == w && _imgVp[3] == h)
      return;

  _imgVp = _image->getViewport();

  std::vector<GLfloat> points(w*h*3);

  for(int y = 0; y<h; y++)
  {
    for(int x = 0; x<w; x++)
    {
      points[y*w*3+x*3]   = static_cast<float>(x);
      points[y*w*3+x*3+1] = static_cast<float>(y);
      points[y*w*3+x*3+2] = 0.f;
    }
  }

  // VBO for points
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(GLfloat), &points[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void vvIbrClient::createThreads()
{
  vvDebugMsg::msg(1, "vvIbrClient::createThreads()");

  _thread = new Thread(2);

  pthread_create(&_thread->thread, NULL, getImageFromSocket, this);

  pthread_barrier_wait(&_thread->startBarrier);
}

void vvIbrClient::destroyThreads()
{
  vvDebugMsg::msg(1, "vvIbrClient::destroyThreads()");

  pthread_cancel(_thread->thread);
  pthread_join(_thread->thread, NULL);

  delete _thread;
}

void* vvIbrClient::getImageFromSocket(void* threadargs)
{
  vvDebugMsg::msg(1, "vvIbrClient::getImageFromSocket(): thread started");

  vvIbrClient *ibr = static_cast<vvIbrClient*>(threadargs);
  vvIbrImage* img = ibr->_image;

  pthread_barrier_wait(&ibr->_thread->startBarrier);

  while (ibr->_socketIO)
  {
    pthread_mutex_lock( &ibr->_thread->imageMutex );
    pthread_cond_wait(&ibr->_thread->imageCond, &ibr->_thread->imageMutex);
    vvSocket::ErrorType err = ibr->_socketIO->getIbrImage(img);
    if(err != vvSocket::VV_OK)
    {
      std::cerr << "vvIbrClient::getImageFromSocket: socket-error (" << err << ") - exiting..." << std::endl;
      pthread_mutex_unlock( &ibr->_thread->imageMutex );
      break;
    }
    img->decode();
    pthread_mutex_unlock( &ibr->_thread->imageMutex );
    //vvToolshed::sleep(1000);

    pthread_mutex_lock( &ibr->_thread->signalMutex );
    ibr->_newFrame = true;
    ibr->_haveFrame = true;
    pthread_cond_signal(&ibr->_thread->readyCond);
    pthread_mutex_unlock( &ibr->_thread->signalMutex );
  }
  pthread_exit(NULL);

  vvDebugMsg::msg(1, "vvIbrClient::getImageFromSocket(): thread terminated");
#if !defined(__GNUC__) || defined(__MINGW32__)
  return NULL;
#endif
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
