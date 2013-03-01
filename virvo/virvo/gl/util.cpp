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


#include "util.h"

#include <GL/glew.h>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <stdio.h>
#include <stdarg.h>


namespace gl = virvo::gl;


#ifndef APIENTRY
#define APIENTRY
#endif


#ifndef NDEBUG


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static char const* GetDebugTypeString(GLenum type)
{
    switch (type)
    {
    case GL_DEBUG_TYPE_ERROR:
        return "error";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        return "deprecated behavior detected";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        return "undefined behavior detected";
    case GL_DEBUG_TYPE_PORTABILITY:
        return "portablility warning";
    case GL_DEBUG_TYPE_PERFORMANCE:
        return "performance warning";
    case GL_DEBUG_TYPE_OTHER:
        return "other";
    case GL_DEBUG_TYPE_MARKER:
        return "marker";
    }

    return "{unknown type}";
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
#ifdef _WIN32
static void OutputDebugStringAF(char const* format, ...)
{
    char text[1024] = {0};

    va_list args;
    va_start(args, format);

    vsnprintf_s(text, _TRUNCATE, format, args);

    va_end(args);

    OutputDebugStringA(text);
}
#endif

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void APIENTRY DebugCallback( GLenum /*source*/,
                                    GLenum type,
                                    GLuint /*id*/,
                                    GLenum /*severity*/,
                                    GLsizei /*length*/,
                                    const GLchar* message,
                                    GLvoid* /*userParam*/
                                    )
{
    if (type != GL_DEBUG_TYPE_ERROR)
        return;

#ifdef _WIN32
    if (IsDebuggerPresent())
    {
        OutputDebugStringAF("GL %s: %s\n", GetDebugTypeString(type), message);
        DebugBreak();
    }
#endif

    fprintf(stderr, "GL %s: %s\n", GetDebugTypeString(type), message);
}


#endif // !NDEBUG


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::enableDebugCallback()
{
#ifndef NDEBUG
    if (GLEW_KHR_debug)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

        glDebugMessageCallback(DebugCallback, 0);
    }
#endif // !NDEBUG
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
GLenum gl::getError(char const* file, int line)
{
    GLenum err = glGetError();

    if (err != GL_NO_ERROR)
        fprintf(stderr, "%s(%d) : GL error: %s\n", file, line, gluErrorString(err));

    return err;
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static char const* GetFramebufferStatusString(GLenum status)
{
    switch (status)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        return "framebuffer complete";

    case GL_FRAMEBUFFER_UNDEFINED:
        // is returned if target is the default framebuffer, but the default
        // framebuffer does not exist.
        return "framebuffer undefined";

    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        // is returned if any of the framebuffer attachment points are
        // framebuffer incomplete.
        return "framebuffer incomplete attachment";

    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        // is returned if the framebuffer does not have at least one image
        // attached to it.
        return "framebuffer incomplete missing attachment";

    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        // is returned if the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE
        // is GL_NONE for any color attachment point(s) named by GL_DRAWBUFFERi.
        return "framebuffer incomplete draw buffer";

    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        // is returned if GL_READ_BUFFER is not GL_NONE and the value of
        // GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for the color
        // attachment point named by GL_READ_BUFFER.
        return "framebuffer incomplete read buffer";

    case GL_FRAMEBUFFER_UNSUPPORTED:
        // is returned if the combination of internal formats of the attached
        // images violates an implementation-dependent set of restrictions.
        return "framebuffer unsupported";

    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        // is returned if the value of GL_RENDERBUFFER_SAMPLES is not the same
        // for all attached renderbuffers; if the value of GL_TEXTURE_SAMPLES is
        // the not same for all attached textures; or, if the attached images
        // are a mix of renderbuffers and textures, the value of
        // GL_RENDERBUFFER_SAMPLES does not match the value of
        // GL_TEXTURE_SAMPLES.
        // GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE  is also returned if the value
        // of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not the same for all attached
        // textures; or, if the attached images are a mix of renderbuffers and
        // textures, the value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not
        // GL_TRUE for all attached textures.
        return "framebuffer incomplete multisample";

    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        // is returned if any framebuffer attachment is layered, and any
        // populated attachment is not layered, or if all populated color
        // attachments are not from textures of the same target.
        return "framebuffer incomplete layer targets";

    default:
        return "{unknown status}";
    }
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
GLenum gl::getFramebufferStatus(GLenum target, char const* file, int line)
{
    GLenum status = glCheckFramebufferStatus(target);

    if (status != GL_FRAMEBUFFER_COMPLETE)
        fprintf(stderr, "%s(%d) : GL framebuffer error: %s\n", file, line, GetFramebufferStatusString(status));

    return status;
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void DrawFullScreenQuad()
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::blendTexture(GLuint texture, GLenum sfactor, GLenum dfactor)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glActiveTexture(GL_TEXTURE0);

    glEnable(GL_BLEND);
    glBlendFunc(sfactor, dfactor);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    glDepthMask(GL_FALSE);

    DrawFullScreenQuad();

    glPopAttrib();
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::blendTexture(GLuint texture)
{
    blendTexture(texture, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::blendPixels(GLsizei srcW, GLsizei srcH, GLenum format, GLenum type, const GLvoid* pixels, GLenum sfactor, GLenum dfactor)
{
    glPushAttrib(GL_COLOR_BUFFER_BIT);

    GLint viewport[4];

    glGetIntegerv(GL_VIEWPORT, &viewport[0]);

    glWindowPos2i(viewport[0], viewport[1]);

    GLfloat scaleX = static_cast<GLfloat>(viewport[2]) / srcW;
    GLfloat scaleY = static_cast<GLfloat>(viewport[3]) / srcH;

    glPixelZoom(scaleX, scaleY);

    glEnable(GL_BLEND);
    glBlendFunc(sfactor, dfactor);

    glDrawPixels(srcW, srcH, format, type, pixels);

    glPopAttrib();
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::blendPixels(GLsizei srcW, GLsizei srcH, GLenum format, GLenum type, const GLvoid* pixels)
{
    blendPixels(srcW, srcH, format, type, pixels, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::renderInterlacedStereoStencilBuffer(bool lines)
{
    static const GLubyte kPatternLines[32*(32/8)] = {
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
    };

    static const GLubyte kPatternCheckerBoard[32*(32/8)] = {
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
    };

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_LIGHTING);

    glColorMask(0, 0, 0, 0);
    glDepthMask(0);

    glClearStencil(0);
    glClear(GL_STENCIL_BUFFER_BIT);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glEnable(GL_POLYGON_STIPPLE);
    if (lines)
        glPolygonStipple(kPatternLines);
    else
        glPolygonStipple(kPatternCheckerBoard);

    glEnable(GL_STENCIL_TEST);
    glStencilMask(0xFFFFFFFF);
    glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
    glStencilFunc(GL_ALWAYS, 1, 0xFFFFFFFF);

    DrawFullScreenQuad();

    glPopAttrib();
    glPopClientAttrib();
}
