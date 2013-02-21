// handle.cpp


#include "handle.h"

#include <GL/glew.h>


namespace gl = virvo::gl;


void gl::Buffer::destroy()
{
    glDeleteBuffers(1, &name);
}


void gl::Framebuffer::destroy()
{
    glDeleteFramebuffers(1, &name);
}


void gl::Renderbuffer::destroy()
{
    glDeleteRenderbuffers(1, &name);
}


void gl::Texture::destroy()
{
    glDeleteTextures(1, &name);
}


unsigned gl::createBuffer()
{
    unsigned n = 0;
    glGenBuffers(1, &n);
    return n;
}


unsigned gl::createFramebuffer()
{
    unsigned n = 0;
    glGenFramebuffers(1, &n);
    return n;
}


unsigned gl::createRenderbuffer()
{
    unsigned n = 0;
    glGenRenderbuffers(1, &n);
    return n;
}


unsigned gl::createTexture()
{
    unsigned n = 0;
    glGenTextures(1, &n);
    return n;
}
