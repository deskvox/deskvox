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
#include "vvopengl.h"

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvgltools.h"

using namespace std;

//============================================================================
// Method Definitions
//============================================================================

//----------------------------------------------------------------------------
/** Check OpenGL for errors.
    @param  error string if there was an error, otherwise return NULL
*/
void vvGLTools::printGLError(const char* msg)
{
  GLenum err = glGetError();
  if(err != GL_NO_ERROR)
  {
    char* str = (char*)gluErrorString(err);
    cerr << "GL error: " << msg << ", " << str << endl;
  }
}

//----------------------------------------------------------------------------
/** Checks OpenGL for a specific extension.
    @param extension OpenGL extension to check for (e.g. "GL_EXT_bgra")
    @return true if extension is supported
*/
bool vvGLTools::isGLextensionSupported(const char* extension)
{
  const GLubyte *extensions = NULL;
  const GLubyte *start;
  GLubyte *where, *terminator;

  // Check requested extension name for existence and for spaces:
  where = (GLubyte*)strchr(extension, ' ');
  if (where || *extension=='\0') return false;

  // Get extensions string from OpenGL:
  extensions = glGetString(GL_EXTENSIONS);
  if (extensions=='\0') return false;

  // Parse OpenGL extensions string:
  start = extensions;
  for (;;)
  {
    where = (GLubyte*)strstr((const char*)start, extension);
    if (!where) return false;
    terminator = where + strlen(extension);
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
void vvGLTools::displayOpenGLextensions(DisplayStyle style)
{
  char* extensions = NULL;                        // OpenGL extensions string
  char* extCopy;                                  // local copy of extensions string for modifications
  int i;

  extensions = (char*)glGetString(GL_EXTENSIONS);

  switch (style)
  {
    default:
    case CONSECUTIVE:
      cerr << extensions << endl;
      break;
    case ONE_BY_ONE:
      extCopy = new char[strlen(extensions) + 1];
      strcpy(extCopy, extensions);
      for (i=0; i<(int)strlen(extCopy); ++i)
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
  const char* status[3] = {"supported", "not found"};

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

//============================================================================
// End of File
//============================================================================
