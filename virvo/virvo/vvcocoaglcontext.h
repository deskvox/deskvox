#ifndef VV_COCOAGLCONTEXT_H
#define VV_COCOAGLCONTEST_H

#ifdef __APPLE__

#ifdef __OBJC__
class vvContextOptions;
#else
class vvContextOptions;
class NSAutoreleasePool;
class NSOpenGLContext;
class NSOpenGLPixelFormat;
class NSView;
class NSWindow;
#endif

class vvCocoaGLContext
{
public:
  vvCocoaGLContext(const vvContextOptions& options);
  ~vvCocoaGLContext();

  bool makeCurrent() const;
  void swapBuffers() const;
  void resize(int w, int h);
private:
  const vvContextOptions& _options;

  NSAutoreleasePool* _autoreleasePool;
  NSOpenGLContext* _context;
  NSOpenGLPixelFormat* _pixelFormat;
  NSWindow* _win;
  NSView* _glView;

  void init();
  void destroy();
  void createGLContext();
};

#endif

#endif

