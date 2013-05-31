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

#include <iostream>

#include <string.h>
#include <cstdlib>
#include <GL/glew.h>
#include "vvopengl.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvgltools.h"

#include <sstream>

using namespace std;

namespace
{
  bool debugAbortOnGLError = false;
  bool debugPrintBacktrace = false;

  /** Callback function for GL errors.
    If the extension GL_ARB_debug_output is available, this callback
    function will be called automatically if a GL error is generated
    */
#ifndef WINAPI
#define WINAPI
#endif
  void WINAPI debugCallback(GLenum /*source*/, GLenum /*type*/, GLuint /*id*/, GLenum /*severity*/,
      GLsizei /*length*/, GLchar const* message, GLvoid* /*userParam*/)
  {
    std::cerr << "GL error: " << message << std::endl;
    if(debugPrintBacktrace)
      vvToolshed::printBacktrace();

    if(debugAbortOnGLError)
      abort();
  }
}

//============================================================================
// Method Definitions
//============================================================================

bool vvGLTools::enableGLErrorBacktrace(bool printBacktrace, bool abortOnError)
{
// As of January 2011, only freeglut supports glutInitContextFlags
// with GLUT_DEBUG. This may be outdated in the meantime as may be
// those checks!

  debugPrintBacktrace = printBacktrace;
  debugAbortOnGLError = abortOnError;

#ifdef GL_ARB_debug_output
#if !defined(__GNUC__) || !defined(_WIN32)
  glewInit();
  if (glDebugMessageCallbackARB != NULL)
  {
    cerr << "Init callback function for GL_ARB_debug_output extension" << endl;
    glDebugMessageCallbackARB(debugCallback, NULL);
    return true;
  }
  else
#endif
  {
    cerr << "glDebugMessageCallbackARB not available" << endl;
    return false;
  }
#else
  cerr << "Consider installing GLEW >= 1.5.7 for extension GL_ARB_debug_output" << endl;
  return false;
#endif // GL_ARB_debug_output
}

//----------------------------------------------------------------------------
/** Check OpenGL for errors.
    @param  error string if there was an error, otherwise return NULL
*/
void vvGLTools::printGLError(const char* msg)
{
  const GLenum err = glGetError();
  if(err != GL_NO_ERROR)
  {
    const char* str = (const char*)gluErrorString(err);
    cerr << "GL error: " << msg << ", " << str << endl;
  }
}

vvGLTools::GLInfo vvGLTools::getGLInfo()
{
  GLInfo result;

  result.vendor = (const char*)glGetString(GL_VENDOR);
  result.renderer = (const char*)glGetString(GL_RENDERER);
  result.version = (const char*)glGetString(GL_VERSION);

  return result;
}

//----------------------------------------------------------------------------
/** Checks OpenGL for a specific OpenGL version.
    @param major OpenGL major version to check
    @param minor OpenGL minor version to check
    @param release OpenGL release version to check
    @return true if version is supported
*/
bool vvGLTools::isGLVersionSupported(int major, int minor, int release)
{
  (void)release;
  // Get version string from OpenGL:
  const GLubyte* verstring = glGetString(GL_VERSION);
  if (verstring=='\0') return false;

  int ver[3] = { 0, 0, 0 };
  int idx = 0;
  for (const GLubyte *p = verstring;
      *p && *p != ' ' && idx < 3;
      ++p)
  {
    if (*p == '.')
    {
      ++idx;
    }
    else if (*p >= '0' && *p <= '9')
    {
      ver[idx] *= 10;
      ver[idx] += *p-'0';
    }
    else
      return false;
  }

  vvDebugMsg::msg(3, "GL version ", ver[0], ver[1], ver[2]);

  if(ver[0] < major)
    return false;
  if(ver[0] > major)
    return true;

  if(ver[1] < minor)
    return false;
  if(ver[1] >= minor)
    return true;

  return false;
}

//----------------------------------------------------------------------------
/** Checks OpenGL for a specific extension.
    @param extension OpenGL extension to check for (e.g. "GL_EXT_bgra")
    @return true if extension is supported
*/
bool vvGLTools::isGLextensionSupported(const char* extension)
{
  // Check requested extension name for existence and for spaces:
  const GLubyte* where = (GLubyte*)strchr(extension, ' ');
  if (where || *extension=='\0') return false;

  // Get extensions string from OpenGL:
  const GLubyte* extensions = glGetString(GL_EXTENSIONS);
  if (extensions=='\0') return false;

  // Parse OpenGL extensions string:
  const GLubyte* start = extensions;
  for (;;)
  {
    where = (GLubyte*)strstr((const char*)start, extension);
    if (!where) return false;
    const GLubyte* terminator = where + strlen(extension);
    if (where==start || *(where - 1)==' ')
      if (*terminator==' ' || *terminator=='\0')
        return true;
    start = terminator;
  }
}

//----------------------------------------------------------------------------
/** Display the OpenGL extensions which are supported by the system at
  run time.
  @param style display style
*/
void vvGLTools::displayOpenGLextensions(const DisplayStyle style)
{
  char* extCopy;                                  // local copy of extensions string for modifications

  const char* extensions = (const char*)glGetString(GL_EXTENSIONS);

  switch (style)
  {
    default:
    case CONSECUTIVE:
      cerr << extensions << endl;
      break;
    case ONE_BY_ONE:
      extCopy = new char[strlen(extensions) + 1];
      strcpy(extCopy, extensions);
      for (int i=0; i<(int)strlen(extCopy); ++i)
        if (extCopy[i] == ' ') extCopy[i] = '\n';
      cerr << extCopy << endl;
      delete[] extCopy;
      break;
  }
}

//----------------------------------------------------------------------------
/** Check for some specific OpenGL extensions.
  Displays the status of volume rendering related extensions, each on a separate line.
*/
void vvGLTools::checkOpenGLextensions()
{
  const char* status[2] = {"supported", "not found"};

  cerr << "GL_EXT_texture3D...............";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_texture3D")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_texture_edge_clamp......";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_texture_edge_clamp")) ? status[0] : status[1]) << endl;

  cerr << "GL_SGI_texture_color_table.....";
  cerr << ((vvGLTools::isGLextensionSupported("GL_SGI_texture_color_table")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_paletted_texture........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_paletted_texture")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_blend_equation..........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_blend_equation")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_shared_texture_palette..";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_shared_texture_palette")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_blend_minmax............";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax")) ? status[0] : status[1]) << endl;

  cerr << "GL_ARB_multitexture............";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ARB_multitexture")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_texture_shader...........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_texture_shader")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_texture_shader2..........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_texture_shader2")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_texture_shader3..........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_texture_shader3")) ? status[0] : status[1]) << endl;

  cerr << "GL_ARB_texture_env_combine.....";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ARB_texture_env_combine")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_register_combiners.......";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_register_combiners")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_register_combiners2......";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_register_combiners2")) ? status[0] : status[1]) << endl;

  cerr << "GL_ARB_fragment_program........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ARB_fragment_program")) ? status[0] : status[1]) << endl;

  cerr << "GL_ATI_fragment_shader.........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ATI_fragment_shader")) ? status[0] : status[1]) << endl;

  cerr << "GL_ARB_imaging.................";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ARB_imaging")) ? status[0] : status[1]) << endl;
}


//----------------------------------------------------------------------------
/** Draw view aligned quad. If no vertex coordinates are specified,
    these default to: (-1.0f, -1.0f) (1.0f, 1.0f). No multi texture coordinates
    supported.
*/
void vvGLTools::drawQuad(float x1, float y1, float x2, float y2)
{
  glBegin(GL_QUADS);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glNormal3f(0.0f, 0.0f, 1.0f);

    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(x1, y1);

    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(x2, y1);

    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(x2, y2);

    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(x1, y2);
  glEnd();
}

//----------------------------------------------------------------------------
/** Get OpenGL viewport info: 0 ==> x, 1 ==> y, 2 ==> width, 3 ==> height.
*/
virvo::Viewport vvGLTools::getViewport()
{
  virvo::Viewport result;
  glGetIntegerv(GL_VIEWPORT, result.values);
  return result;
}

//----------------------------------------------------------------------------
/** Query the color specificied using glClearColor (rgba)
*/
vvVector4 vvGLTools::queryClearColor()
{
  GLfloat tmp[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, tmp);
  return vvVector4(tmp[0], tmp[1], tmp[2], tmp[3]);
}

//----------------------------------------------------------------------------
/** Get the current modelview matrix.
  @param a matrix which will be set to the current modelview matrix
*/
void vvGLTools::getModelviewMatrix(vvMatrix* mv)
{
  GLfloat glmatrix[16];                           // OpenGL compatible matrix

  vvDebugMsg::msg(3, "vvRenderer::getModelviewMatrix()");
  glGetFloatv(GL_MODELVIEW_MATRIX, glmatrix);
  mv->setGL((float*)glmatrix);
}

//----------------------------------------------------------------------------
/** Get the current projection matrix.
  @param a matrix which will be set to the current projection matrix
*/
void vvGLTools::getProjectionMatrix(vvMatrix* pm)
{
  vvDebugMsg::msg(3, "vvRenderer::getProjectionMatrix()");

  GLfloat glmatrix[16];                           // OpenGL compatible matrix
  glGetFloatv(GL_PROJECTION_MATRIX, glmatrix);
  pm->setGL((float*)glmatrix);
}

//----------------------------------------------------------------------------
/** Set the OpenGL modelview matrix.
    Adjusts the opengl matrix mode to GL_MODELVIEW.
  @param new OpenGL modelview matrix
*/
void vvGLTools::setModelviewMatrix(const vvMatrix& mv)
{
  vvDebugMsg::msg(3, "vvRenderer::setModelviewMatrix()");

  GLfloat glmatrix[16];                           // OpenGL compatible matrix
  mv.getGL((float*)glmatrix);
  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(glmatrix);
}

//----------------------------------------------------------------------------
/** Set the OpenGL projection matrix.
    Adjusts the opengl matrix mode to GL_PROJECTION.
  @param new OpenGL projection matrix
*/
void vvGLTools::setProjectionMatrix(const vvMatrix& pm)
{
  vvDebugMsg::msg(3, "vvRenderer::setProjectionMatrix()");

  GLfloat glmatrix[16];                           // OpenGL compatible matrix
  pm.getGL((float*)glmatrix);
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(glmatrix);
}

//----------------------------------------------------------------------------
/** Calc near- and far clipping in eye coordinates
  @param znear Near-clipping plane
  @param zfar  Far-clipping plane
*/
void vvGLTools::getClippingPlanes(vvPlane& znear, vvPlane& zfar)
{
  // Normalized device coordinates.
  znear = vvPlane(vvVector3(0.0f, 0.0f, -1.0f), vvVector3(0.0f, 0.0f, 1.0f));
  zfar = vvPlane(vvVector3(0.0f, 0.0f, 1.0f), vvVector3(0.0f, 0.0f, 1.0f));

  vvMatrix invPr;
  vvGLTools::getProjectionMatrix(&invPr);
  invPr.invert();

  // Matrix to transform normal.
  vvMatrix trPr;
  vvGLTools::getProjectionMatrix(&trPr);
  //trPr.invert();
  trPr.transpose();
  //trPr.invert();

  // Transform to eye coordinates.
  znear._point.multiply(invPr);
  zfar._point.multiply(invPr);
  znear._normal.multiply(trPr);
  zfar._normal.multiply(trPr);
  znear._normal.normalize();
  zfar._normal.normalize();
}

//----------------------------------------------------------------------------
/** Return a projected vertex (gluProject).
  @param obj coordinate
*/
vvVector3 vvGLTools::project(const vvVector3& obj)
{
  double modelview[16];
  double projection[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  glGetDoublev(GL_PROJECTION_MATRIX, projection);
  virvo::Viewport viewport = vvGLTools::getViewport();
  double winX;
  double winY;
  double winZ;

  gluProject(obj[0], obj[1], obj[2],
             modelview, projection, viewport.values,
             &winX, &winY, &winZ);
  return vvVector3(static_cast<float>(winX),
                   static_cast<float>(winY),
                   static_cast<float>(winZ));
}

//----------------------------------------------------------------------------
/** Return an un-projected vertex (gluUnProject).
  @param win coordinate
*/
vvVector3 vvGLTools::unProject(const vvVector3& win)
{
  double modelview[16];
  double projection[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  glGetDoublev(GL_PROJECTION_MATRIX, projection);
  virvo::Viewport viewport = vvGLTools::getViewport();
  double objX;
  double objY;
  double objZ;

  gluUnProject(win[0], win[1], win[2],
               modelview, projection, viewport.values,
               &objX, &objY, &objZ);
  return vvVector3(static_cast<float>(objX),
                   static_cast<float>(objY),
                   static_cast<float>(objZ));
}

std::string virvo::gltools::lastError(const std::string& file, int line)
{
  std::stringstream out;
  const GLenum err = glGetError();
  if(err != GL_NO_ERROR)
  {
    std::string str(reinterpret_cast<const char*>(gluErrorString(err)));
    out << file << ":" << line << ": OpenGL error: " << str;
  }
  return out.str();
}

virvo::Matrix virvo::gltools::getModelViewMatrix()
{
  Matrix mv;
  vvGLTools::getModelviewMatrix(&mv);
  return mv;
}

virvo::Matrix virvo::gltools::getProjectionMatrix()
{
  Matrix pr;
  vvGLTools::getProjectionMatrix(&pr);
  return pr;
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
