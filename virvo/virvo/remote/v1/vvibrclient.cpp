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

#include <GL/glew.h>

#include "vvibrclient.h"
#include "vvibr.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvsocketio.h"
#include "vvdebugmsg.h"
#include "vvvoldesc.h"
#include "vvpthread.h"

#include "private/vvgltools.h"
#include "private/vvibrimage.h"

#include "gl/util.h"

#include <memory>
#include <stdexcept>

using std::cerr;
using std::endl;

struct vvIbrClient::Impl
{
  // OpenGL client state flags
  struct GLState
  {
    GLState()
      : colorArray(false)
      , edgeFlagArray(false)
      , indexArray(false)
      , normalArray(false)
      , textureCoordArray(false)
      , vertexArray(false)
    {
    }
    GLboolean colorArray;
    GLboolean edgeFlagArray;
    GLboolean indexArray;
    GLboolean normalArray;
    GLboolean textureCoordArray;
    GLboolean vertexArray;
  };
  GLState glstate;

  void pushGLClientState()
  {
    glstate.colorArray        = glIsEnabled(GL_COLOR_ARRAY);
    glstate.edgeFlagArray     = glIsEnabled(GL_EDGE_FLAG_ARRAY);
    glstate.indexArray        = glIsEnabled(GL_INDEX_ARRAY);
    glstate.normalArray       = glIsEnabled(GL_NORMAL_ARRAY);
    glstate.textureCoordArray = glIsEnabled(GL_TEXTURE_COORD_ARRAY);
    glstate.vertexArray       = glIsEnabled(GL_VERTEX_ARRAY);
  }

  void popGLClientState()
  {
    glstate.colorArray ? glEnableClientState(GL_COLOR_ARRAY) : glDisableClientState(GL_COLOR_ARRAY);
    glstate.edgeFlagArray ? glEnableClientState(GL_EDGE_FLAG_ARRAY) : glDisableClientState(GL_EDGE_FLAG_ARRAY);
    glstate.indexArray ? glEnableClientState(GL_INDEX_ARRAY) : glDisableClientState(GL_INDEX_ARRAY);
    glstate.normalArray ? glEnableClientState(GL_NORMAL_ARRAY) : glDisableClientState(GL_NORMAL_ARRAY);
    glstate.textureCoordArray ? glEnableClientState(GL_TEXTURE_COORD_ARRAY) : glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glstate.vertexArray ? glEnableClientState(GL_VERTEX_ARRAY) : glDisableClientState(GL_VERTEX_ARRAY);
  }
};

struct vvIbrClient::Thread
{
  pthread_t thread;               ///< list for threads of each server connection
  virvo::Mutex mutex;
  virvo::SyncedCondition imageRequest;
  bool cancel;

  Thread()
    : cancel(false)
  {
  }
};

vvIbrClient::vvIbrClient(vvVolDesc *vd, vvRenderState renderState,
                         vvTcpSocket* socket, const std::string& filename)
  : vvRemoteClient(vd, renderState, socket, filename)
  , impl_(new Impl)
  , _thread(new Thread)
  , _shader(vvShaderFactory().createProgram("ibr", "", "ibr"))
{
  vvDebugMsg::msg(1, "vvIbrClient::vvIbrClient()");

  rendererType = REMOTE_IBR;
  _socketIO->putEvent(virvo::RemoteServerType);
  _socketIO->putRendererType(rendererType);

  glewInit();

  if (!GLEW_VERSION_1_5)
    throw std::runtime_error("OpenGL 1.5 or later required");

  if (!_shader)
    throw std::runtime_error("Could not find ibr shaders");

  glGenBuffers(1, &_pointVBO);
  glGenTextures(1, &_rgbaTex);
  glGenTextures(1, &_depthTex);

  createThreads();
}

vvIbrClient::~vvIbrClient()
{
  vvDebugMsg::msg(1, "vvIbrClient::~vvIbrClient()");

  destroyThreads();

  glDeleteBuffers(1, &_pointVBO);
  glDeleteTextures(1, &_rgbaTex);
  glDeleteTextures(1, &_depthTex);

  delete _shader;
}

vvRemoteClient::ErrorType vvIbrClient::render()
{
  vvDebugMsg::msg(1, "vvIbrClient::render()");

  // Request a new image
  this->_thread->imageRequest.signal();

  bool imageNew = false;

  {
    virvo::ScopedLock lock(&this->_thread->mutex);

    imageNew = this->_nextImage.get() != 0;
    if (imageNew)
    {
      this->_image.reset(this->_nextImage.release());
    }
  }

  // Check for a valid image
  if (this->_image.get() == 0 || this->_image->width() <= 0 || this->_image->height() <= 0)
    return VV_OK;

  // If this is a new image, update the textures and buffers
  if (imageNew)
    initIbrFrame();

  // Draw boundary lines
  if (_boundaries)
  {
    const vvVector3 size(vd->getSize()); // volume size [world coordinates]
    drawBoundingBox(size, vd->pos, _boundColor);
  }

  vvMatrix currentMatrix = _currentPr * _currentMv;

  float drMin = 0.0f;
  float drMax = 0.0f;
  vvAABB aabb = vvAABB(vvVector3(), vvVector3());
  vd->getBoundingBox(aabb);
  vvIbr::calcDepthRange(_currentPr, _currentMv, aabb, drMin, drMax);
  const virvo::Viewport vp = vvGLTools::getViewport();
  vvMatrix currentImgMatrix = vvIbr::calcImgMatrix(_currentPr, _currentMv, vp, drMin, drMax);
  bool matrixChanged = (!currentImgMatrix.equal(_imgMatrix));

  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

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

  _shader->setParameter1f("vpWidth", static_cast<float>(vp[2]));
  _shader->setParameter1f("vpHeight", static_cast<float>(vp[3]));
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
/*  float v_i[16] = {
    0.70710678f, 0.70710678f, 0.0f, 0.0f,
   -0.70710678f, 0.70710678f, 0.0f, 0.0f,
           0.0f,        0.0f, 1.0f, 0.0f,
           0.0f,        0.0f, 0.0f, 1.0f
  };*/
  float v_i[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  };
  vvMatrix V_i = vvMatrix(v_i);
  V_i.invert();

  _shader->setParameter1f("si", 1.0);
  _shader->setParameter1f("sj", 1.0);
  _shader->setParameter1f("sk", 1.0);
  _shader->setParameterMatrix4f("V_i" , V_i);
  //// end ellipsoid test code - temporary

  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

  // push the client state and temporarily disable every
  // client state but vertex arrays
  impl_->pushGLClientState();
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_EDGE_FLAG_ARRAY);
  glDisableClientState(GL_INDEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, NULL);

  glDrawArrays(GL_POINTS, 0, _imgVp[2]*_imgVp[3]);

  glDisableClientState(GL_VERTEX_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  impl_->pushGLClientState();

  _shader->disable();

  if(depthMask)
    glDepthMask(GL_TRUE);
  if(!pointSmooth)
    glDisable(GL_POINT_SMOOTH);

  return VV_OK;
}

void vvIbrClient::initIbrFrame()
{
  vvDebugMsg::msg(1, "vvIbrClient::initIbrFrame()");

  const int h = _image->height();
  const int w = _image->width();

  _imgPr = _image->projMatrix();
  _imgMv = _image->viewMatrix();
  _imgDepthRange[0] = _image->depthMin();
  _imgDepthRange[1] = _image->depthMax();
  _imgMatrix = vvIbr::calcImgMatrix(_imgPr, _imgMv, _image->viewport(), _imgDepthRange[0], _imgDepthRange[1]);

  // get pixel and depth-data
  virvo::PixelFormatInfo cf = mapPixelFormat(_image->colorBuffer().format());

  glBindTexture(GL_TEXTURE_2D, _rgbaTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, cf.internalFormat, w, h, 0, cf.format, cf.type, _image->colorBuffer().data().ptr());

  glBindTexture(GL_TEXTURE_2D, _depthTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  virvo::PixelFormatInfo df = mapPixelFormat(_image->depthBuffer().format());

  glTexImage2D(GL_TEXTURE_2D, 0, df.internalFormat, w, h, 0, df.format, df.type, _image->depthBuffer().data().ptr());

  if(_imgVp[2] == w && _imgVp[3] == h)
      return;

  _imgVp = _image->viewport();

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

  pthread_create(&_thread->thread, NULL, getImageFromSocket, this);
}

void vvIbrClient::destroyThreads()
{
  vvDebugMsg::msg(1, "vvIbrClient::destroyThreads()");

  this->_thread->cancel = true;
  this->_thread->imageRequest.signal(); // wake up the thread in case it's waiting...

  pthread_join(_thread->thread, NULL);
}

#define THREAD_FAILURE ((void*)-1)

void* vvIbrClient::getImageFromSocket(void* threadargs)
{
  vvDebugMsg::msg(1, "vvIbrClient::getImageFromSocket(): thread started");

  vvIbrClient *ibr = static_cast<vvIbrClient*>(threadargs);

  for (;;)
  {
    // Block until a new request arrives...
    ibr->_thread->imageRequest.wait();

    // Exit?
    if (ibr->_thread->cancel)
      break;

    // Send the request
    if (VV_OK != ibr->requestFrame())
      return THREAD_FAILURE;

    // Create a new image
    std::auto_ptr<virvo::IbrImage> image(new virvo::IbrImage);

    // Get the image
    if (vvSocket::VV_OK != ibr->_socketIO->getIbrImage(*image))
      return THREAD_FAILURE;

    // Decompress the image
    if (!image->decompress())
      return THREAD_FAILURE;

    virvo::ScopedLock lock(&ibr->_thread->mutex);

    ibr->_nextImage.reset(image.release()); // Swap the images
  }

  return 0;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
