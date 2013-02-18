// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2012 University of Stuttgart, 2004-2005 Brown University
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


#ifndef VV_GL_HANDLE_H
#define VV_GL_HANDLE_H


#include <GL/glew.h>

#include <assert.h>
#include <string>

#include "private/vvcompiler.h"


// Makes subclasses of Handle<T> non-copyable, but moveable
#ifndef CXX_NO_RVALUE_REFERENCES

#define VV_GL_MAKE_MOVE_ONLY(CLASS)                     \
    private:                                            \
        typedef Handle<CLASS> BaseType;                 \
                                                        \
        CLASS(CLASS const&);                            \
        CLASS& operator =(CLASS const&);                \
    public:                                             \
        CLASS(CLASS&& rhs) : BaseType(std::move(rhs))   \
        {                                               \
        }                                               \
        CLASS& operator =(CLASS&& rhs)                  \
        {                                               \
            BaseType::operator =(std::move(rhs));       \
            return *this;                               \
        }

#else

#define VV_GL_MAKE_MOVE_ONLY(CLASS)                     \
    private:                                            \
        typedef Handle<CLASS> BaseType;                 \
                                                        \
        CLASS(CLASS const&);                            \
        CLASS& operator =(CLASS const&);                \
    public:

#endif


namespace virvo
{
namespace gl
{


    //----------------------------------------------------------------------------------------------
    // Handle
    //
    // Base class for all OpenGL objects.
    // Manages the lifetime of the currently held object.
    //

    template<class Derived>
    class Handle
    {
    protected:
        GLuint handle;

    public:
        // Construct from the given object
        explicit Handle(GLuint handle) : handle(handle) {}

#ifndef CXX_NO_RVALUE_REFERENCES
        // Move construct from another handle
        Handle(Handle&& RHS) : handle(RHS.handle)
        {
            RHS.handle = 0;
        }
#endif

#ifndef CXX_NO_RVALUE_REFERENCES
        // Move assign from another handle
        Handle& operator =(Handle&& RHS)
        {
            reset(RHS.handle);

            RHS.handle = 0;

            return *this;
        }
#endif

        // Clean up:
        // Destroy the OpenGL object
        ~Handle() { reset(); }

        // Returns the OpenGL handle
        GLuint get() const { return handle; }

        // Reset with another OpenGL object (of the same type!!!)
        void reset(GLuint h = 0)
        {
            if (handle)
                static_cast<Derived*>(this)->destroy();

            handle = h;
        }

        // Release ownership of the OpenGL object
        GLuint release()
        {
            GLuint h = handle;
            handle = 0;
            return h;
        }

    private:
        // NOT copyable!
        Handle(Handle const&);
        Handle& operator =(Handle const&);
    };


    //----------------------------------------------------------------------------------------------
    // Buffer
    //

    class Buffer : public Handle<Buffer>
    {
        VV_GL_MAKE_MOVE_ONLY(Buffer)

    public:
        // Construct from the given object
        explicit Buffer(GLuint buffer = 0) : Handle<Buffer>(buffer) {}

        // Destroy the buffer object
        void destroy() { glDeleteBuffers(1, &this->handle); }
    };


    //----------------------------------------------------------------------------------------------
    // Renderbuffer
    //

    class Renderbuffer : public Handle<Renderbuffer>
    {
        VV_GL_MAKE_MOVE_ONLY(Renderbuffer)

    public:
        // Construct from the given object
        explicit Renderbuffer(GLuint renderbuffer = 0) : Handle<Renderbuffer>(renderbuffer) {}

        // Destroy the renderbuffer object
        void destroy() { glDeleteRenderbuffers(1, &this->handle); }
    };


    //----------------------------------------------------------------------------------------------
    // Texture
    //

    class Texture : public Handle<Texture>
    {
        VV_GL_MAKE_MOVE_ONLY(Texture)

    public:
        // Construct from the given object
        explicit Texture(GLuint texture = 0) : Handle<Texture>(texture) {}

        // Destroy the texture object
        void destroy() { glDeleteTextures(1, &this->handle); }
    };


    //----------------------------------------------------------------------------------------------
    // Shader
    //

#if 0
    class Shader : public Handle<Shader>
    {
        VV_GL_MAKE_MOVE_ONLY(Shader)

    public:
        // Construct from the given object
        explicit Shader(GLuint shader = 0) : Handle(shader) {}

        // Destroy the shader object
        void destroy() { glDeleteShader(this->handle); }

        // Returns a shader parameter
        GLint getParameter(GLenum pname) const;

        // Sets the shader source
        void setSource(char const* source);

        // Sets the shader source
        void setSourceFromFile(char const* filename);

        // Compiles the shader object
        void compile();

        // Returns whether the last call to compile() was successful
        bool isCompiled() const;

        // Returns the shader type
        GLenum getType() const;

        // Returns the shader info-log
        std::string getInfoLog() const;
    };
#endif


    //----------------------------------------------------------------------------------------------
    // Free functions
    //

    // Creates an OpenGL buffer
    inline GLuint createBuffer()
    {
        GLuint h = 0;

        glGenBuffers(1, &h);

        return h;
    }

    // Creates an OpenGL renderbuffer
    inline GLuint createRenderbuffer()
    {
        GLuint h = 0;

        glGenRenderbuffers(1, &h);

        return h;
    }

    // Creates an OpenGL texture
    inline GLuint createTexture()
    {
        GLuint h = 0;

        glGenTextures(1, &h);

        return h;
    }

#if 0

    // Creates an OpenGL shader
    inline GLuint createShader(GLenum type)
    {
        return glCreateShader(type);
    }

    // Creates an OpenGL shader
    inline GLuint createShader(GLenum type, char const* source)
    {
        Shader shader(glCreateShader(type));

        shader.setSource(source);
        shader.compile();

        return shader.release();
    }

    // Creates an OpenGL shader
    inline GLuint createShaderFromFile(GLenum type, char const* filename)
    {
        Shader shader(glCreateShader(type));

        shader.setSourceFromFile(filename);
        shader.compile();

        return shader.release();
    }

#endif


} // namespace gl
} // namespace virvo


#undef VV_GL_MAKE_MOVE_ONLY


#endif // VV_GL_HANDLE_H
