//****************************************************************************
// Project:         Virvo (Virtual Reality Volume Renderer)
// Copyright:       (c) 1999-2004 Jurgen P. Schulze. All rights reserved.
// Author's E-Mail: schulze@cs.brown.edu
// Affiliation:     Brown University, Department of Computer Science
//****************************************************************************

#ifndef _VVVIEW_
#define _VVVIEW_

/**
 * Virvo File Viewer main class.
 * The Virvo File Viewer is a quick alternative to the Java based VEdit
 * environment. It can display all Virvo file types using any implemented
 * algorithm, but it has only limited information and transfer function edit
 * capabilites.<P>
 * Usage:
 * <UL>
 *   <LI>Accepts the volume filename as a command line argument
 *   <LI>Mouse moved while left button pressed: rotate
 *   <LI>Mouse moved while middle button pressed: translate
 *   <LI>Mouse moved while right button pressed and menus are off: scale
 * </UL>
 *
 *  This program supports the following macro definitions at compile time:
 * <DL>
 *   <DT>VV_DICOM_SUPPORT</DT>
 *   <DD>If defined, the Papyrus library is used and DICOM files can be read.</DD>
 *   <DT>VV_VOLUMIZER_SUPPORT</DT>
 *   <DD>If defined, SGI Volumizer is supported for rendering (only on SGIs).</DD>
 * </DL>
 *
 * @author Juergen Schulze (schulze@cs.brown.de)
 */

class vvStopwatch;

class vvView
{
   private:
      /// Mouse buttons (to be or'ed for multiple buttons pressed simultaneously)
      enum
      {
         NO_BUTTON     = 0,                       ///< no button pressed
         LEFT_BUTTON   = 1,                       ///< left button pressed
         MIDDLE_BUTTON = 2,                       ///< middle button pressed
         RIGHT_BUTTON  = 4                        ///< right button pressed
      };
      enum                                        ///  Timer callback types
      {
         ANIMATION_TIMER = 0,                     ///< volume animation timer callback
         ROTATION_TIMER  = 1                      ///< rotation animation timer callback
      };
      static const int ROT_TIMER_DELAY;           ///< rotation timer delay in milliseconds
      static const int DEFAULTSIZE;               ///< default window size (width and height) in pixels
      static const float OBJ_SIZE;                ///< default object size
      static const int DEFAULT_PORT;              ///< default port for socket connections
      static vvView* ds;                          ///< one instance of VView is always present
      vvObjView* ov;                              ///< the current view on the object
      vvRenderer* renderer;                       ///< rendering engine
      vvRenderState renderState;                  ///< renderer state
      vvVolDesc* vd;                              ///< volume description
      char* filename;                             ///< volume file name
      int   window;                               ///< GLUT window handle
      int   winWidth, winHeight;                  ///< window size in pixels
      int   pressedButton;                        ///< ID of currently pressed button
      int   lastX, lastY;                         ///< previous mouse coordinates
      int   curX, curY;                           ///< current mouse coordinates
      int   x1,y1,x2,y2;                          ///< mouse coordinates for auto-rotation
      int   lastWidth, lastHeight;                ///< last window size
      int   lastPosX, lastPosY;                   ///< last window position
      bool  emptySpaceLeapingMode;                ///< true = bricks invisible due to current transfer function aren't rendered
      bool  perspectiveMode;                      ///< true = perspective projection
      bool  boundariesMode;                       ///< true = display boundaries
      bool  orientationMode;                      ///< true = display axis orientation
      bool  fpsMode;                              ///< true = display fps
      bool  paletteMode;                          ///< true = display transfer function palette
      int   stereoMode;                           ///< 0=mono, 1=active stereo, 2=passive stereo (views side by side)
      bool  activeStereoCapable;                  ///< true = hardware is active stereo capable
      bool  interpolMode;                         ///< true = linear interpolation in slices
      bool  preintMode;                           ///< true = use pre-integration
      bool  opCorrMode;                           ///< true = do opacity correction
      bool  gammaMode;                            ///< true = do gamma correction
      int   mipMode;                              ///< 1 = maximum intensity projection, 2=minimum i.p.
      bool  fullscreenMode;                       ///< true = full screen mode enabled
      bool  timingMode;                           ///< true = display rendering times in text window
      bool  menuEnabled;                          ///< true = popup menu is enabled
      int   mainMenu;                             ///< ID of main menu
      bool  animating;                            ///< true = animation mode on
      bool  rotating;                             ///< true = rotation mode on
      bool  rotationMode;                         ///< true = auto-rotation possible
      int   frame;                                ///< current animation frame
      vvTexRend::GeometryType currentGeom;        ///< current rendering geometry
      vvTexRend::VoxelType currentVoxels;         ///< current voxel type
      float bgColor[3];                           ///< background color (R,G,B in [0..1])
      float draftQuality;                         ///< current draft mode rendering quality (>0)
      float highQuality;                          ///< current high quality mode rendering quality (>0)
      bool  hqMode;                               ///< true = high quality mode on, false = draft quality
      bool  refinement;                           ///< true = use high/draft quality modes, false = always use draft mode
      const char* onOff[2];                       ///< strings for "on" and "off"
      float warpMatrix[16];                       ///< warp matrix for 2D image transfer
      vvVector3 pos;                              ///< volume position in object space
      float animSpeed;                            ///< time per animation frame
      bool  iconMode;                             ///< true=display file icon
      const char** displayNames;                  ///< X-window display names to use (relevant for threaded rendering)
      unsigned int numDisplays;                   ///< number of displays to use
      bool gpuproxygeo;                           ///< true=compute proxy geometry on gpu
      bool useOffscreenBuffer;                    ///< render to an offscreen buffer. Mandatory for setting buffer precision
      int  bufferPrecision;                       ///< 8 or 32 bit. Higher res can minimize rounding error during slicing

   public:
      vvView();
      ~vvView();
      int run(int, char**);

   private:
      static void reshapeCallback(int, int);
      static void displayCallback();
      static void buttonCallback(int, int, int, int);
      static void motionCallback(int, int);
      static void keyboardCallback(unsigned char, int, int);
      static void specialCallback(int, int, int);
      static void timerCallback(int);
      static void mainMenuCallback(int);
      static void rendererMenuCallback(int);
      static void voxelMenuCallback(int);
      static void optionsMenuCallback(int);
      static void transferMenuCallback(int);
      static void animMenuCallback(int);
      static void viewMenuCallback(int);
      static void performanceTest();
      static void printProfilingInfo(const int testNr = 1, const int testCnt = 1);
      static void printProfilingResult(vvStopwatch* totalTime, const int framesRendered);
      void setAnimationFrame(int);
      void initGraphics(int argc, char *argv[]);
      void createMenus();
      void setRenderer(vvTexRend::GeometryType=vvTexRend::VV_AUTO, vvTexRend::VoxelType=vvTexRend::VV_BEST);
      void setProjectionMode(bool);
      void displayHelpInfo();
      bool parseCommandLine(int argc, char *argv[]);
      void addDisplay(const char* name);
      void mainLoop(int argc, char *argv[]);
};
#endif
