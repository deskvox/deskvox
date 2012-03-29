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

void vvCocoaGLContext::init()
{
  _autoreleasePool = [[NSAutoreleasePool alloc] init];

  (void)[NSApplication sharedApplication];
  NSRect rect = NSMakeRect(0.0f, 0.0f,
                           static_cast<float>(_options->width),
                           static_cast<float>(_options->height));
  NSWindow* win = [[NSWindow alloc]
    initWithContentRect:rect
    styleMask: NSTitledWindowMask
    backing: NSBackingStoreBuffered
    defer:NO];

  if (!win)
  {
    std::cerr << "Couldn't open NSWindow" << std::endl;
  }
  [win makeKeyAndOrderFront:nil];

  NSRect glRect = NSMakeRect(0.0f, 0.0f,
                             static_cast<float>(_options->width),
                             static_cast<float>(_options->height));

  NSView* glView = [[NSView alloc] initWithFrame:glRect];
  [win setContentView:glView];
  createGLContext();
  [_context setView:glView];
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

