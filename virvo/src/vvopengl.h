#ifndef VV_OPENGL_H
#define VV_OPENGL_H

#include "vvexport.h"

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__)
#define GL_GLEXT_PROTOTYPES 1
#ifndef GL_GLEXT_LEGACY
# define GL_GLEXT_LEGACY 1
#endif
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#else
#include <windows.h>
#endif
#endif

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "vvglext.h"
#endif
