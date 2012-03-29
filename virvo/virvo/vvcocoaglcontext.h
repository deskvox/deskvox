#ifndef VV_COCOAGLCONTEXT_H
#define VV_COCOAGLCONTEST_H

#ifdef __APPLE__

class vvRenderContext;
class vvContextOptions;
class NSAutoreleasePool;
class NSOpenGLContext;
class NSOpenGLPixelFormat;

class vvCocoaGLContext
{
public:
  vvCocoaGLContext(vvContextOptions* options);
  ~vvCocoaGLContext();

  bool makeCurrent() const;
  void swapBuffers() const;
private:
  vvContextOptions* _options;

  NSAutoreleasePool* _autoreleasePool;
  NSOpenGLContext* _context;
  NSOpenGLPixelFormat* _pixelFormat;

  void init();
  void createGLContext();
};

#endif

#endif

