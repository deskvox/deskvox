#ifndef VV_OPENGL_H
#define VV_OPENGL_H

#include "vvexport.h"

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__)
#define GL_GLEXT_PROTOTYPES 1
#ifndef GL_GLEXT_LEGACY
# define GL_GLEXT_LEGACY 1
#endif
#endif

#include "vvplatform.h"

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#ifndef _WIN32
#include <GL/glx.h>
#endif
#endif

#endif
