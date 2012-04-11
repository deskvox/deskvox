#ifndef VV_COCOAGLCONTEXT_H
#define VV_COCOAGLCONTEST_H

#ifdef __APPLE__

class vvRenderContext;
struct vvContextOptions;
class NSAutoreleasePool;
class NSOpenGLContext;
class NSOpenGLPixelFormat;
class NSView;
class NSWindow;

class vvCocoaGLContext
{
public:
  vvCocoaGLContext(vvContextOptions* options);
  ~vvCocoaGLContext();

  bool makeCurrent() const;
  void swapBuffers() const;
  void resize(int w, int h);
private:
  vvContextOptions* _options;

  NSAutoreleasePool* _autoreleasePool;
  NSOpenGLContext* _context;
  NSOpenGLPixelFormat* _pixelFormat;
  NSWindow* _win;
  NSView* _glView;

  void init();
  void createGLContext();
};

#endif

#endif

