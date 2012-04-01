#import <Cocoa/Cocoa.h>
#import <iostream>

#import "vvcocoaglcontext.h"
#import "vvrendercontext.h"

vvCocoaGLContext::vvCocoaGLContext(vvContextOptions* options)
  : _options(options)
{
  init();
}

vvCocoaGLContext::~vvCocoaGLContext()
{
  [_autoreleasePool release];
  _autoreleasePool = 0;
}

bool vvCocoaGLContext::makeCurrent() const
{
  if (_context != NULL)
  {
    [_context makeCurrentContext];
    return true;
  }
  else
  {
    return false;
  }
}

void vvCocoaGLContext::swapBuffers() const
{
  [_context flushBuffer];
}

void vvCocoaGLContext::resize(const int w, const int h)
{
  if (_win != NULL)
  {
    NSRect rect = NSMakeRect(0.0f, 0.0f,
                             static_cast<float>(w),
                             static_cast<float>(h));
    [_win setFrame: rect display: YES];
    [_glView setFrame: rect];
    [_context update];
  }
}

void vvCocoaGLContext::init()
{std::cerr << _options->width << std::endl;
  _autoreleasePool = [[NSAutoreleasePool alloc] init];

  (void)[NSApplication sharedApplication];
  NSRect rect = NSMakeRect(0.0f, 0.0f,
                           static_cast<float>(_options->width),
                           static_cast<float>(_options->height));
  _win = [[NSWindow alloc]
    initWithContentRect:rect
    styleMask: NSTitledWindowMask
    backing: NSBackingStoreBuffered
    defer:NO];

  if (!_win)
  {
    std::cerr << "Couldn't open NSWindow" << std::endl;
  }
  [_win makeKeyAndOrderFront:nil];

  NSRect glRect = NSMakeRect(0.0f, 0.0f,
                             static_cast<float>(_options->width),
                             static_cast<float>(_options->height));

  _glView = [[NSView alloc] initWithFrame:glRect];
  [_win setContentView:_glView];
  createGLContext();
  [_context setView:_glView];
  [_context update];
  makeCurrent();
}

void vvCocoaGLContext::createGLContext()
{
  NSOpenGLPixelFormatAttribute attr[] = { 
    NSOpenGLPFAAccelerated,
    NSOpenGLPFADepthSize,
    (NSOpenGLPixelFormatAttribute)32,
    _options->doubleBuffering ? NSOpenGLPFADoubleBuffer : (NSOpenGLPixelFormatAttribute)nil,
    (NSOpenGLPixelFormatAttribute)nil
  };

  _pixelFormat = (NSOpenGLPixelFormat*)[[NSOpenGLPixelFormat alloc]
    initWithAttributes: attr];

  _context = (NSOpenGLContext*)[[NSOpenGLContext alloc]
    initWithFormat: _pixelFormat
    shareContext: nil];
}

