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

#include <iostream>
#include <iomanip>

#include <float.h>
#ifdef _WIN32
#include <windows.h>
#elif _LINUX64BIT
#include <dlfcn.h>
#else
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #include <dirent.h>
#endif
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvtoolshed.h"

#ifdef __sun
#define powf pow
#define atanf atan
#define sqrtf sqrt
#define sinf sin
#define cosf cos
#endif

using namespace std;

#ifdef __hpux
# include <sys/pstat.h>
# include <sys/param.h>
# include <sys/unistd.h>
#endif
//#define VV_STANDALONE      // define to perform self test

int vvToolshed::progressSteps = 0;

//============================================================================
// Method Definitions
//============================================================================

//----------------------------------------------------------------------------
/** Case insensitive string comparison
    @param str1,str2 pointers to strings that are being compared
    @return
      <UL>
        <LI> 0 if equal
        <LI>-1 if str1<str2
        <LI> 1 if str1>str2
      </UL>
*/
int vvToolshed::strCompare(const char* str1, const char* str2)
{
#ifdef _WIN32
  return stricmp(str1, str2);
#else
  return strcasecmp(str1, str2);
#endif
}

//----------------------------------------------------------------------------
/** Case insensitive string comparison with a number of characters.
    @param str2,str2 pointers to strings that are being compared
    @param n = number of characters to compare
    @return the same values as in #strCompare(const char*, const char*)
*/
int vvToolshed::strCompare(const char* str1, const char* str2, int n)
{
#ifdef _WIN32
  return strnicmp(str1, str2, n);
#else
  return strncasecmp(str1, str2, n);
#endif
}

//----------------------------------------------------------------------------
/** Case insensitive string suffix comparison.
    @param str    pointer to string
    @param suffix pointer to suffix
    @return true if suffix is the suffix of str
*/
bool vvToolshed::isSuffix(const char* str, const char* suffix)
{
  if (vvToolshed::strCompare(str + strlen(str) - strlen(suffix), suffix) == 0)
    return true;
  else return false;
}

//----------------------------------------------------------------------------
/** Convert HSB color model to RGB.
    @param a hue (0..360) (becomes red)
    @param b saturation (0..1) (becomes green)
    @param c value = brightness (0..1) (becomes blue)
    @return RGB values in a,b,c
*/
void vvToolshed::HSBtoRGB(float* a, float* b, float* c)
{
  float red, green, blue;

  HSBtoRGB(*a, *b, *c, &red, &green, &blue);
  *a = red;
  *b = green;
  *c = blue;
}

//----------------------------------------------------------------------------
/** Convert HSB color model to RGB.
    @param h hue (0..1)
    @param s saturation (0..1)
    @param v value = brightness (0..1)
    @return RGB values (0..1) in r,g,b
*/
void vvToolshed::HSBtoRGB(float h, float s, float v, float* r, float* g, float* b)
{
  float f, p, q, t;
  int i;

  // Clamp values to their valid ranges:
  h = ts_clamp(h, 0.0f, 1.0f);
  s = ts_clamp(s, 0.0f, 1.0f);
  v = ts_clamp(v, 0.0f, 1.0f);

  // Convert hue:
  if (h == 1.0f) h = 0.0f;
  h *= 360.0f;

  if (s==0.0f)                                    // grayscale value?
  {
    *r = v;
    *g = v;
    *b = v;
  }
  else
  {
    h /= 60.0;
    i = int(h);
    f = h - i;
    p = v * (1.0f - s);
    q = v * (1.0f - (s * f));
    t = v * (1.0f - (s * (1.0f - f)));
    switch (i)
    {
      case 0: *r = v; *g = t; *b = p; break;
      case 1: *r = q; *g = v; *b = p; break;
      case 2: *r = p; *g = v; *b = t; break;
      case 3: *r = p; *g = q; *b = v; break;
      case 4: *r = t; *g = p; *b = v; break;
      case 5: *r = v; *g = p; *b = q; break;
    }
  }
}

//----------------------------------------------------------------------------
/** Convert RGB colors to HSB model
    @param a red [0..360] (becomes hue)
    @param b green [0..1] (becomes saturation)
    @param c blue [0..1]  (becomes brightness)
    @return HSB in a,b,c
*/
void vvToolshed::RGBtoHSB(float* a, float* b, float* c)
{
  float h,s,v;
  RGBtoHSB(*a, *b, *c, &h, &s, &v);
  *a = h;
  *b = s;
  *c = v;
}

//----------------------------------------------------------------------------
/** Convert RGB colors to HSB model.
    @param r,g,b RGB values [0..1]
    @return h = hue [0..1], s = saturation [0..1], v = value = brightness [0..1]
*/
void vvToolshed::RGBtoHSB(float r, float g, float b, float* h, float* s, float* v)
{
  float max, min, delta;

  // Clamp input values to valid range:
  r = ts_clamp(r, 0.0f, 1.0f);
  g = ts_clamp(g, 0.0f, 1.0f);
  b = ts_clamp(b, 0.0f, 1.0f);

  max = ts_max(r, ts_max(g, b));
  min = ts_min(r, ts_min(g, b));
  *v = max;
  *s = (max != 0.0f) ? ((max - min) / max) :0.0f;
  if (*s == 0.0f) *h = 0.0f;
  else
  {
    delta = max - min;
    if (r==max)
      *h = (g - b) / delta;
    else if (g==max)
      *h = 2.0f + (b - r) / delta;
    else if (b==max)
      *h = 4.0f + (r - g) / delta;
    *h *= 60.0f;
    if (*h < 0.0f)
      *h += 360.0f;
  }
  *h /= 360.0f;
}

//----------------------------------------------------------------------------
/** Copies the tail string after the last occurrence of a given character.
    Example: str="c:\ local\ testfile.dat", c='\' => suffix="testfile.dat"
    @param suffix <I>allocated</I> space for the found string
    @param str    source string
    @param c      character after which to copy characters
    @return result in suffix, empty string if c was not found in str
*/
void vvToolshed::strcpyTail(char* suffix, const char* str, char c)
{
  int i, j;

  // Search for c in pathname:
  i = strlen(str) - 1;
  while (i>=0 && str[i]!=c)
    --i;

  // Extract tail string:
  if (i<0)                                        // c not found?
  {
    //strcpy(suffix, str);
    strcpy(suffix, "");
  }
  else
  {
    for (j=i+1; j<(int)strlen(str); ++j)
      suffix[j-i-1] = str[j];
    suffix[j-i-1] = '\0';
  }
}

//----------------------------------------------------------------------------
/** Copies the tail string after the last occurrence of a given character.
    Example: str="c:\local\testfile.dat", c='\' => suffix="testfile.dat"
    @param str    source string
    @param c      character after which to copy characters
    @return string after c ("testfile.dat")
*/
string vvToolshed::strcpyTail(const string str, char c)
{
  return str.substr(str.rfind(c) + 1);
}

//----------------------------------------------------------------------------
/** Copies the head string before the first occurrence of a given character.
    Example: str="c:\ local\ testfile.dat", c='.' => head="c:\ local\ testfile"
    @param head  <I>allocated</I> space for the found string
    @param str    source string
    @param c      character before which to copy characters
    @return result in head, empty string if c was not found in str
*/
void vvToolshed::strcpyHead(char* head, const char* str, char c)
{
  int i = 0;

  if (strchr(str, c) == NULL)
  {
    head[0] = '\0';
    return;
  }
  while (str[i] != c)
  {
    head[i] = str[i];
    ++i;
  }
  head[i] = '\0';
}

//----------------------------------------------------------------------------
/** Removes leading and trailing spaces from a string.
    Example: str="  hello " => str="hello"
    @param str    string to trim
    @return result in str
*/
void vvToolshed::strTrim(char* str)
{
  int i;

  // Trim trailing spaces:
  for (i=strlen(str)-1; i>0; --i)
  {
    if (str[i]==' ') str[i] = '\0';
    else break;
  }
  if (str[0]=='\0') return;                       // done

  // Trim leading spaces:
  i=0;
  while (str[i]==' ')
  {
    ++i;
  }
  if (i==0) return;                               // done
  strcpy(str, str+i);
}

//----------------------------------------------------------------------------
/** Extracts a filename from a given path.
    Directory elements have to be separated by '/' or '\' depending on OS.
    @param filename <I>allocated</I> space for filename (e.g. "testfile.dat")
    @param pathname file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in filename
*/
void vvToolshed::extractFilename(char* filename, const char* pathname)
{
#ifdef _WIN32
  char delim = '\\';
#else
  char delim = '/';
#endif

  if (strchr(pathname, delim)) strcpyTail(filename, pathname, delim);
  else strcpy(filename, pathname);
}

//----------------------------------------------------------------------------
/** Extracts a filename from a given path.
    Directory elements have to be separated by '/' or '\' depending on OS.
    @param pathname file including entire path (e.g. "/usr/local/testfile.dat")
    @return filename (e.g. "testfile.dat")
*/
string vvToolshed::extractFilename(const string pathname)
{
#ifdef _WIN32
  char delim = '\\';
#else
  char delim = '/';
#endif

  if (pathname.find(delim, 0) != string::npos) return strcpyTail(pathname, delim);
  else return pathname;
}

//----------------------------------------------------------------------------
/** Extracts a directory name from a given path.
    Directory elements have to be separated by '/' or '\' depending on OS.
    @param dirname  <I>allocated</I> space for directory name (e.g. "/usr/local/" or "c:\user\")
    @param pathname file including entire path (e.g. "/usr/local/testfile.dat" or "c:\user\testfile.dat")
    @return result in dirname
*/
void vvToolshed::extractDirname(char* dirname, const char* pathname)
{
  int i, j;

#ifdef _WIN32
  char delim = '\\';
#else
  char delim = '/';
#endif

  // Search for '\' or '/' in pathname:
  i = strlen(pathname) - 1;
  while (i>=0 && pathname[i]!=delim)
    --i;

  // Extract preceding string:
  if (i<0)                                        // delimiter not found?
    strcpy(dirname, "");
  else
  {
    for (j=0; j<=i; ++j)
      dirname[j] = pathname[j];
    dirname[j] = '\0';
  }
}

//----------------------------------------------------------------------------
/** Extracts a directory name from a given path.
    Directory elements have to be separated by '/' or '\' depending on OS.
    @param pathname file including entire path (e.g. "/usr/local/testfile.dat" or "c:\user\testfile.dat")
    @return directory namename (e.g. "/usr/local/" or "c:\user\")
*/
string vvToolshed::extractDirname(const string pathname)
{
  string dirname;
  size_t delimPos;

#ifdef _WIN32
  char delim = '\\';
#else
  char delim = '/';
#endif

  delimPos = pathname.rfind(delim);
  if (delimPos == string::npos) dirname = pathname;
  else dirname.insert(0, pathname, 0, delimPos+1);
  return dirname;
}

//----------------------------------------------------------------------------
/** Extracts an extension from a given path or filename.
    @param extension <I>allocated</I> space for extension (e.g. "dat")
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in extension
*/
void vvToolshed::extractExtension(char* extension, const char* pathname)
{
  char *filename = new char[strlen(pathname)+1];
  extractFilename(filename, pathname);

  strcpyTail(extension, filename, '.');
  delete[] filename;
}

//----------------------------------------------------------------------------
/** Extracts an extension from a given path or filename.
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return extension, e.g., "dat"
*/
string vvToolshed::extractExtension(const string pathname)
{
  return strcpyTail(pathname, '.');
}

//----------------------------------------------------------------------------
/** Extracts the base file name from a given path or filename, excluding
    the '.' delimiter.
    @param basename  <I>allocated</I> memory space for basename (e.g. "testfile").
                     Memory must be allocated for at least strlen(pathname)+1 chars!
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in basename
*/
void vvToolshed::extractBasename(char* basename, const char* pathname)
{
  int i;

  extractFilename(basename, pathname);

  // Search for '.' in pathname:
  i = strlen(basename) - 1;
  while (i>=0 && basename[i]!='.')
    --i;

  if (i>0) basename[i] = '\0';                    // convert '.' to '\0' to terminate string
}

//----------------------------------------------------------------------------
/** Extracts the base file name from a given path or filename, excluding
    the '.' delimiter.
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return basename (e.g. "testfile").
*/
string vvToolshed::extractBasename(const string pathname)
{
  string basename;

  basename = extractFilename(pathname);
  basename.erase(basename.rfind('.'));
  return basename;
}

//----------------------------------------------------------------------------
/** Remove the extension from a path string. If no '.' is present in path
    string, the path is removed without changes.
    @param basepath  <I>allocated</I> space for path without extension
                     (e.g., "/usr/local/testfile")
    @param pathname  file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in basepath
*/
void vvToolshed::extractBasePath(char* basepath, const char* pathname)
{
  int i, j;

  // Search for '.' in pathname:
  i = strlen(pathname) - 1;
  while (i>=0 && pathname[i]!='.')
    --i;

  // Extract tail string:
  if (i<0)                                        // '.' not found?
  {
    strcpy(basepath, pathname);
  }
  else
  {
    for (j=0; j<i; ++j)
      basepath[j] = pathname[j];
    basepath[j] = '\0';
  }
}

//----------------------------------------------------------------------------
/** Replaces a file extension with a new one, overwriting the old one.
    If the pathname does not have an extension yet, the new extension will be
    added.
    @param newPath      _allocated space_ for resulting path name with new
                        extension (e.g. "/usr/local/testfile.txt")
    @param newExtension new extension without '.' (e.g. "txt")
    @param pathname     file including entire path (e.g. "/usr/local/testfile.dat")
    @return result in newPath
*/
void vvToolshed::replaceExtension(char* newPath, const char* newExtension, const char* pathname)
{
  char* pointPos;
  int baseNameLen;                                // length of base file name, including point

  pointPos = (char*)strrchr(pathname, '.');
  if (pointPos==NULL)                             // is there a point in pathname?
  {
    // No point, so just add new extension:
    strcpy(newPath, pathname);
    strcat(newPath, ".");
    strcat(newPath, newExtension);
  }
  else
  {
    baseNameLen = pointPos-pathname+1;
    memcpy(newPath, pathname, baseNameLen);       // copy everything before the point, including the point
    newPath[baseNameLen] = '\0';
    strcat(newPath, newExtension);
  }
}

//----------------------------------------------------------------------------
/** Increases the filename (filename must include an extension!).
  @return true if successful, false if filename couldn't be increased.
          Does not check if the file with the increased name exists.
*/
bool vvToolshed::increaseFilename(char* filename)
{
  bool done = false;
  int i;
  char ext[256];

  extractExtension(ext, filename);
  if (strlen(ext)==0) i=strlen(filename) - 1;
  else i = strlen(filename) - strlen(ext) - 2;
  while (!done)
  {
    if (i<0 || filename[i]<'0' || filename[i]>'9')
      return false;

    if (filename[i] == '9')                       // overflow?
    {
      filename[i] = '0';
      --i;
    }
    else
    {
      ++filename[i];
      done = 1;
    }
  }
  return true;
}

//----------------------------------------------------------------------------
/** Increases the filename (filename must include an extension!).
  @return true if successful, false if filename couldn't be increased.
          Does not check if the file with the increased name exists.
*/
bool vvToolshed::increaseFilename(string& filename)
{
  bool done = false;
  int i;
  string ext;

  ext = extractExtension(filename);
  if (ext.size()==0) i = filename.size() - 1;
  else i = filename.size() - ext.size() - 2;
  while (!done)
  {
    if (i<0 || filename[i]<'0' || filename[i]>'9')
      return false;

    if (filename[i] == '9')                       // overflow?
    {
      filename[i] = '0';
      --i;
    }
    else
    {
      ++filename[i];
      done = 1;
    }
  }
  return true;
}

//----------------------------------------------------------------------------
/** Draws a line in a 3D volume dataset using Bresenham's algorithm.
    Both line end points must lie within the volume. The Coordinate system is:
    <PRE>
           y
           |__ x
          /
         z
    </PRE>
    The volume data is arranged like this:
    <UL>
      <LI>origin is top left front
<LI>width in positive x direction
<LI>height in negative y direction
<LI>slices in negative z direction
</UL>
@param x0,y0,z0  line starting point in voxels
@param x1,y1,z1  line end point in voxels
@param color     array with line color elements (size = bpm * mod)
@param data      pointer to raw volume data
@param bpv       bytes per voxel
@param w,h,s     width/height/slices of volume data array [voxels]
*/
void vvToolshed::draw3DLine(int x0, int y0, int z0, int x1, int y1, int z1,
uchar* color, uchar* data, int bytes, int w, int h, int s)
{
  int xd, yd, zd;
  int x, y, z;
  int ax, ay, az;
  int sx, sy, sz;
  int dx, dy, dz;
  int i;

  x0 = ts_clamp(x0, 0, w-1);
  x1 = ts_clamp(x1, 0, w-1);
  y0 = ts_clamp(y0, 0, h-1);
  y1 = ts_clamp(y1, 0, h-1);
  z0 = ts_clamp(z0, 0, s-1);
  z1 = ts_clamp(z1, 0, s-1);

  dx = x1 - x0;
  dy = y1 - y0;
  dz = z1 - z0;

  ax = ts_abs(dx) << 1;
  ay = ts_abs(dy) << 1;
  az = ts_abs(dz) << 1;

  sx = ts_zsgn(dx);
  sy = ts_zsgn(dy);
  sz = ts_zsgn(dz);

  x = x0;
  y = y0;
  z = z0;

  if (ax >= ts_max(ay, az))                       // x is dominant
  {
    yd = ay - (ax >> 1);
    zd = az - (ax >> 1);
    for (;;)
    {
      for (i=0; i<bytes; ++i)
      {
        data[bytes * (z * w * h + y * w + x) + i] = color[i];
      }
      if (x == x1) return;
      if (yd >= 0)
      {
        y += sy;
        yd -= ax;
      }
      if (zd >= 0)
      {
        z += sz;
        zd -= ax;
      }
      x += sx;
      yd += ay;
      zd += az;
    }
  }
  else if (ay >= ts_max(ax, az))                  // y is dominant
  {
    xd = ax - (ay >> 1);
    zd = az - (ay >> 1);
    for (;;)
    {
      for (i=0; i<bytes; ++i)
      {
        data[bytes * (z * w * h + y * w + x) + i] = color[i];
      }
      if (y == y1) return;
      if (xd >= 0)
      {
        x += sx;
        xd -= ay;
      }
      if (zd >= 0)
      {
        z += sz;
        zd -= ay;
      }
      y += sy;
      xd += ax;
      zd += az;
    }
  }
  else if (az >= ts_max(ax, ay))                  // z is dominant
  {
    xd = ax - (az >> 1);
    yd = ay - (az >> 1);
    for (;;)
    {
      for (i=0; i<bytes; ++i)
      {
        data[bytes * (z * w * h + y * w + x) + i] = color[i];
      }
      if (z == z1) return;
      if (xd >= 0)
      {
        x += sx;
        xd -= az;
      }
      if (yd >= 0)
      {
        y += sy;
        yd -= az;
      }
      z += sz;
      xd += ax;
      yd += ay;
    }
  }
}

//----------------------------------------------------------------------------
/** Draws a line in a 2D image dataset using Bresenham's algorithm.
    Both line end points must lie within the image. The coordinate system is:
    <PRE>
           y
           |__ x
    </PRE>
    The image data is arranged like this:
    <UL>
      <LI>origin is top left
      <LI>width is in positive x direction
      <LI>height is in negative y direction
</UL>
@param x0/y0  line starting point in pixels
@param x1/y1  line end point in pixels
@param color  line color, 32 bit value: bits 0..7=first color,
8..15=second color etc.
@param data   pointer to raw image data
@param bpp    byte per pixel (e.g. 3 for 24 bit RGB), range: [1..4]
@param w/h    width/height of image data array in pixels
*/
void vvToolshed::draw2DLine(int x0, int y0, int x1, int y1,
uint color, uchar* data, int bpp, int w, int h)
{
  int xd, yd;
  int x, y;
  int ax, ay;
  int sx, sy;
  int dx, dy;
  int i;
  uchar col[4];                                   // color components; 0=most significant byte

  assert(bpp <= 4);

  col[0] = (uchar)((color >> 24) & 0xff);
  col[1] = (uchar)((color >> 16) & 0xff);
  col[2] = (uchar)((color >> 8)  & 0xff);
  col[3] = (uchar)(color & 0xff);

  x0 = ts_clamp(x0, 0, w-1);
  x1 = ts_clamp(x1, 0, w-1);
  y0 = ts_clamp(y0, 0, h-1);
  y1 = ts_clamp(y1, 0, h-1);

  dx = x1 - x0;
  dy = y1 - y0;

  ax = ts_abs(dx) << 1;
  ay = ts_abs(dy) << 1;

  sx = ts_zsgn(dx);
  sy = ts_zsgn(dy);

  x = x0;
  y = y0;

  if (ax >= ay)                                   // x is dominant
  {
    yd = ay - (ax >> 1);
    for (;;)
    {
      for (i=0; i<bpp; ++i)
        data[bpp * (y * w + x) + i] = col[i];
      if (x == x1) return;
      if (yd >= 0)
      {
        y += sy;
        yd -= ax;
      }
      x += sx;
      yd += ay;
    }
  }
  else if (ay >= ax)                              // y is dominant
  {
    xd = ax - (ay >> 1);
    for (;;)
    {
      for (i=0; i<bpp; ++i)
        data[bpp * (y * w + x) + i] = col[i];
      if (y == y1) return;
      if (xd >= 0)
      {
        x += sx;
        xd -= ay;
      }
      y += sy;
      xd += ax;
    }
  }
}

//----------------------------------------------------------------------------
/** Compute texture hardware compatible numbers.
    @param imgSize  the image size [pixels]
    @return the closest power-of-2 value that is greater than or equal to imgSize.
*/
int vvToolshed::getTextureSize(int imgSize)
{
  return (int)powf(2.0f, (float)ceil(log((float)imgSize) / log(2.0f)));
}

//----------------------------------------------------------------------------
/** Checks if a file exists.
    @param filename file name to check for
    @return true if file exists
*/
bool vvToolshed::isFile(const char* filename)
{
#ifdef _WIN32
  FILE* fp = fopen(filename, "rb");
  if (fp==NULL) return false;
  fclose(fp);
  return true;
#else
  struct stat buf;
  if (stat(filename, &buf) == 0)
  {
    if (S_ISREG(buf.st_mode)) return true;
  }
  return false;
#endif
}

//----------------------------------------------------------------------------
/** Checks if a directory exists.
    @param dir directory name to check for
    @return true if directory exists
*/
bool vvToolshed::isDirectory(const char* path)
{
#ifdef _WIN32
  WIN32_FIND_DATA fileInfo;
  HANDLE found;
  found = FindFirstFile((LPCWSTR)path, &fileInfo);
  bool ret;

  if (found == INVALID_HANDLE_VALUE) return false;
  else
  {
    if(fileInfo.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY) ret = true;
    else ret = false;
    FindClose(found);
    return ret;
  }
#else
  struct stat buf;
  if (stat(path, &buf) == 0)
  {
    if (S_ISDIR(buf.st_mode)) return true;
  }
  return false;
#endif
}

//----------------------------------------------------------------------------
/** Figures out the size of a file in bytes.
    @param  filename file name including path
    @return file size in bytes or -1 on error
*/
long vvToolshed::getFileSize(const char* filename)
{
  FILE* fp;
  long size;

  fp = fopen(filename, "rb");
  if (fp==NULL) return -1;
  if (fseek(fp, 0L, SEEK_END) != 0)
  {
    fclose(fp);
    return -1;
  }
  size = ftell(fp);
  fclose(fp);
  return size;
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in an uchar data array.
    @param data        source data array
    @param elements    number of bytes in source array
    @return minimum and maximum in min and max
*/
void vvToolshed::getMinMax(const uchar* data, int elements, int* min, int* max)
{
  int i;

  *min = 255;
  *max = 0;

  for (i=0; i<elements; ++i)
  {
    if (data[i] > *max) *max = data[i];
    if (data[i] < *min) *min = data[i];
  }
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in a 16 bit big endian data array.
    @param data        source data array
    @param elements    number of 16 bit elements in source array
    @return minimum and maximum in min and max
*/
void vvToolshed::getMinMax16bitBE(const uchar* data, int elements, int* min, int* max)
{
  int i;
  int value;
  int bytes;

  *min = 65535;
  *max = 0;
  bytes = 2 * elements;

  for (i=0; i<bytes; i+=2)
  {
    value = (int(data[i]) << 8) | int(data[i+1]);
    if (value > *max) *max = value;
    if (value < *min) *min = value;
  }
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in an RGBA dataset, only
    considering the alpha component.
    @param data        source data array
    @param elements    number of RGBA elements in source array
    @return minimum and maximum of the alpha component in min and max
*/
void vvToolshed::getMinMaxAlpha(const uchar* data, int elements, int* min, int* max)
{
  int i;
  int bytes;

  *min = 255;
  *max = 0;
  bytes = 4 * elements;

  for (i=3; i<bytes; i+=4)
  {
    if (data[i] > *max) *max = data[i];
    if (data[i] < *min) *min = data[i];
  }
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in a float data array.
    @param data        source data array
    @param elements    number of elements in source array
    @return minimum and maximum in min and max
*/
void vvToolshed::getMinMax(const float* data, int elements, float* min, float* max)
{
  int i;

  *min = FLT_MAX;
  *max = -(*min);

  for (i=0; i<elements; ++i)
  {
    if (data[i] > *max) *max = data[i];
    if (data[i] < *min) *min = data[i];
  }
}

//----------------------------------------------------------------------------
/** Find the minimum and the maximum values in a data array and specify
    a value which is to be ignored, i.e., it does not change the determined
    minimum or maximum values.
    @param data        source data array
    @param elements    number of elements in source array
    @param ignore      value which is to be ignored (e.g, FLT_MAX)
    @return minimum and maximum in min and max
*/
void vvToolshed::getMinMaxIgnore(const float* data, int elements, float ignore,
float* min, float* max)
{
  int i;

  *min = FLT_MAX;
  *max = -(*min);

  for (i=0; i<elements; ++i)
  {
    if (data[i] != ignore)
    {
      if (data[i] > *max) *max = data[i];
      if (data[i] < *min) *min = data[i];
    }
  }
}

//----------------------------------------------------------------------------
/** Convert a sequence of uchar values to float.
    Be sure to have the floatArray allocated before this call!
    @param ucharArray  source array
    @param floatArray  destination array
    @param elements    number of uchar array elements to convert
    @return result in floatArray
*/
void vvToolshed::convertUChar2Float(const uchar* ucharArray, float* floatArray, int elements)
{
  int i;
  for (i=0; i<elements; ++i)
    floatArray[i] = (float)((double)ucharArray[i] / 255.0);
}

//----------------------------------------------------------------------------
/** Convert a sequence of float values to uchar.
    The uchar values cover a range of 0.0 to 1.0 of the float values.
    Be sure to have the ucharArray allocated before this call!
    @param floatArray  source array
    @param ucharArray  destination array
    @param elements    number of float array elements to convert
    @return result in ucharArray
*/
void vvToolshed::convertFloat2UChar(const float* floatArray,
uchar* ucharArray, int elements)
{
  for (int i=0; i<elements; ++i)
    ucharArray[i] = (uchar)(255.0 * floatArray[i]);
}

//----------------------------------------------------------------------------
/** Convert a sequence of float values to uchar.
    The uchar values will cover the range defined by the minimum and
    maximum float values.
    Be sure to have the ucharArray allocated before this call!
    @param floatArray  source array
    @param ucharArray  destination array
    @param elements    number of float array elements to convert
    @param min,max     minimum and maximum float values which will be assigned
                       to 0 and 255 respectively.
    @return result in ucharArray
*/
void vvToolshed::convertFloat2UCharClamp(const float* floatArray,
uchar* ucharArray, int elements, float min, float max)
{
  int i;

  if (min>=max)
    memset(ucharArray, 0, elements);
  else
  {
    for (i=0; i<elements; ++i)
      ucharArray[i] = (uchar)(255.0f * (floatArray[i] - min) / (max - min));
  }
}

//----------------------------------------------------------------------------
/** Convert a sequence of float values to 16 bit values.
    The 16 bit values will cover the range defined by the minimum and
    maximum float values.
    Be sure to have the ucharArray allocated before this call (requires
    elements * 2 bytes)!
    @param floatArray  source array
    @param ucharArray  destination array (16 bit values)
    @param elements    number of float array elements to convert
    @param min,max     minimum and maximum float values which will be assigned
                       to 0 and 65535 respectively.
    @return result in ucharArray
*/
void vvToolshed::convertFloat2ShortClamp(const float* floatArray,
uchar* ucharArray, int elements, float min, float max)
{
  int i;
  int shortValue;

  if (min>=max)
  {
    memset(ucharArray, 0, elements);
  }
  else
  {
    for (i=0; i<elements; ++i)
    {
      shortValue = int(65535.0f * (floatArray[i] - min) / (max - min));
      ucharArray[2*i]   = (uchar)((shortValue >> 8) & 255);
      ucharArray[2*i+1] = (uchar)(shortValue & 255);
    }
  }
}

//----------------------------------------------------------------------------
/** Convert a sequence of float values to uchar.
    The uchar values will cover the range defined by the maximum
    and minimum float values.
    The uchar value of 0 is set for all float values of the 'zero' value,
    and only for them.
    The minimum float value which is not 'zero' becomes an uchar value of 1.
    Be sure to have the ucharArray allocated before this call!
    @param floatArray  source array
    @param ucharArray  destination array
    @param elements    number of float array elements to convert
    @param min,max     minimum and maximum float values which will be assigned
to 1 and 255 respectively.
@param zero        float value which will become 0 in uchar array
@return result in ucharArray
*/
void vvToolshed::convertFloat2UCharClampZero(const float* floatArray,
uchar* ucharArray, int elements, float min, float max, float zero)
{
  int i;

  if (min>=max)
    memset(ucharArray, 0, elements);
  else
    for (i=0; i<elements; ++i)
  {
    if (floatArray[i] == zero)
      ucharArray[i] = (uchar)0;
    else
    {
      if (min==max)
        ucharArray[i] = (uchar)1;
      else
        ucharArray[i] = (uchar)((uchar)(254.0f * (floatArray[i] - min) / (max - min)) + (uchar)1);
    }
  }
}

//----------------------------------------------------------------------------
/** Compute the largest prime factor which is not the number itself.
    @param number number to examine (>1)
    @return largest prime factor, -1 on error
*/
int vvToolshed::getLargestPrimeFactor(int number)
{
  int remainder;
  int factor = 2, largest = 1;

  if (number < 2) return -1;
  remainder = number;
  while (factor < remainder/2)
  {
    if ((remainder % factor) == 0)
    {
      remainder /= factor;
      largest = factor;
    }
    else
      ++factor;
  }
  if (largest==1) return 1;
  else return ts_max(remainder, largest);
}

//----------------------------------------------------------------------------
/** Round the float value to the nearest integer value.
    @param fval value to round
    @return rounded value
*/
int vvToolshed::round(float fval)
{
  return int(fval + 0.5f);
}

//----------------------------------------------------------------------------
/** Initialize progress display.
    @param total total number of progress steps
*/
void vvToolshed::initProgress(int total)
{
  progressSteps = total;
  cerr << "     ";
}

//----------------------------------------------------------------------------
/** Print progress.
  Format: 'xxx %'.
  @param current current progress step
*/
void vvToolshed::printProgress(int current)
{
  int percent, i;

  if (progressSteps<2) percent = 100;
  else
    percent = 100 * current / (progressSteps - 1);
  for (i=0; i<5; ++i)
    cerr << (char)8;                              // ASCII 8 = backspace (BS)
  cerr << setw(3) << percent << " %";
}

//----------------------------------------------------------------------------
/** Run length encode (RLE) a sequence in memory.
  Encoding scheme: X is first data chunk (unsigned char).<UL>
  <LI>if X<128:  copy next X+1 chunks (literal run)</LI>
  <LI>if X>=128: repeat next chunk X-127 times (replicate run)</LI></UL>
  @param out  destination position in memory (must be _allocated_!)
  @param in   source location in memory
  @param size number of bytes to encode
  @param symbol_size  bytes per chunk
  @param space  number of bytes allocated for destination array.
                Encoding process is stopped when this number is reached.
  @return number of bytes written to destination memory or -1 if there is not
enough destination memory, -2 if an invalid data size was passed
@see decodeRLE
@author Michael Poehnl
*/
int vvToolshed::encodeRLE(uchar* out, uchar* in, int size, int symbol_size, int space)
{
  int same_symbol=1;
  int diff_symbol=0;
  int src=0;
  int dest=0;
  bool same;
  int i;

  if ((size % symbol_size) != 0)
  {
    return -2;
  }

  while (src < (size - symbol_size))
  {
    same = true;
    for (i=0; i<symbol_size; i++)
    {
      if (in[src+i] != in[src+symbol_size+i])
      {
        same = false;
        break;
      }
    }
    if (same)
    {
      if (same_symbol == 129)
      {
        assert(dest<space);
        out[dest] = (uchar)(126+same_symbol);
        dest += symbol_size+1;
        same_symbol = 1;
      }
      else
      {
        same_symbol++;
        if (diff_symbol > 0)
        {
          assert(dest<space);
          out[dest] = (uchar)(diff_symbol-1);
          dest += 1+symbol_size*diff_symbol;
          diff_symbol=0;
        }
        if (same_symbol == 2)
        {
          if ((dest+1+symbol_size) > space)
          {
            return -1;
          }
          memcpy(&out[dest+1], &in[src], symbol_size);
        }
      }
    }
    else
    {
      if (same_symbol > 1)
      {
        assert(dest<space);
        out[dest] = (uchar)(126+same_symbol);
        dest += symbol_size+1;
        same_symbol = 1;
      }
      else
      {
        if ((dest+1+diff_symbol*symbol_size+symbol_size) > space)
        {
          return -1;
        }
        memcpy(&out[dest+1+diff_symbol*symbol_size], &in[src], symbol_size);
        diff_symbol++;
        if (diff_symbol == 128)
        {
          assert(dest<space);
          out[dest] = (uchar)(diff_symbol-1);
          dest += 1+symbol_size*diff_symbol;
          diff_symbol=0;
        }
      }
    }
    src += symbol_size;
  }
  if (same_symbol > 1)
  {
    assert(dest<space);
    out[dest] = (uchar)(126+same_symbol);
    dest += symbol_size+1;
  }
  else
  {
    if ((dest+1+diff_symbol*symbol_size+symbol_size) > space)
    {
      return -1;
    }
    memcpy(&out[dest+1+diff_symbol*symbol_size], &in[src], symbol_size);
    diff_symbol++;
    out[dest] = (uchar)(diff_symbol-1);
    dest += 1+symbol_size*diff_symbol;
  }
  return dest;
}

//----------------------------------------------------------------------------
/** Decode a run length encoded (RLE) sequence.
  Data chunks of any byte aligned size can be processed.
  @param out  destination position in memory (_allocated_ space for max bytes)
  @param in   source location in memory
  @param size number of bytes in source array to decode
  @param symbol_size  bytes per chunk (e.g., to encode 24 bit RGB data, use bpc=3)
  @param space  number of allocated bytes in destination memory (for range checking)
  @return number of bytes written to destination memory. If max would
          have been exceeded, -1 is returned
  @see encodeRLE
  @author Michael Poehnl
*/
int vvToolshed::decodeRLE(uchar* out, uchar* in, int size, int symbol_size, int space)
{
  int src=0;
  int dest=0;
  int i, length;

  while (src < size)
  {
    length = (int)in[src];
    if (length > 127)
    {
      for(i=0; i<(length - 126); i++)
      {
        if ((dest + symbol_size) > space)
        {
          return -1;
        }
        memcpy(&out[dest], &in[src+1], symbol_size);
        dest += symbol_size;
      }
      src += 1+symbol_size;
    }
    else
    {
      length++;
      if ((dest + length*symbol_size) > space)
      {
        return -1;
      }
      memcpy(&out[dest], &in[src+1], symbol_size*length);
      dest += length*symbol_size;
      src += 1+symbol_size*length;
    }
  }
  return dest;
}

//----------------------------------------------------------------------------
/** Run length encode (RLE) a sequence of 8 bit values in memory.
  Encoding scheme: X is first data byte (unsigned char).<UL>
  <LI>if X<128:  copy next X+1 bytes (literal run)</LI>
  <LI>if X>=128: repeat next byte X-127 times (replicate run)</LI></UL>
  @param dst  destination position in memory (must be _allocated_!)
  @param src  source location in memory
  @param len  number of bytes to encode
  @param max  number of bytes allocated in destination memory.
              Encoding process is stopped when this number is reached
  @return number of bytes written to destination memory
  @see decodeRLEFast
*/
int vvToolshed::encodeRLEFast(uchar* dst, uchar* src, int len, int max)
{
  int offset;                                     // start position of currently processed run in source array
  int index;                                      // index in source array
  int out;                                        // index in destination array
  int i;                                          // counter
  uchar cur;                                      // currently processed data byte

  offset = out = 0;
  while (offset < len)
  {
    index = offset;
    cur = src[index++];                           // load first data byte from source array
    while (index<len && index-offset<128 && src[index]==cur)
      index++;                                    // search for replicate run
    if (index-offset==1)                          // generate literal run
    {
      // Failed to "replicate" the current byte. See how many to copy.
      // Avoid a replicate run of only 2-pixels after a literal run. There
      // is no gain in this, and there is a risk of loss if the run after
      // the two identical pixels is another literal run. So search for
      // 3 identical pixels.
      while (index<len && index-offset<128 &&
          (src[index]!=src[index-1] || (index>1 && src[index]!=src[index-2])))
        index++;
      // Check why this run stopped. If it found two identical pixels, reset
      // the index so we can add a run. Do this twice: the previous run
      // tried to detect a replicate run of at least 3 pixels. So we may be
      // able to back up two pixels if such a replicate run was found.
      while (index<len && src[index]==src[index-1])
        index--;
      if (out < max)
        dst[out++] = (uchar)(index - offset - 1);
      for (i=offset; i<index; i++)
        if (out < max) dst[out++] = src[i];
    }
    else                                          // generate replicate run
    {
      if (out < max)
        dst[out++] = (uchar)(index - offset + 127);
      if (out < max)
        dst[out++] = cur;
    }
    offset = index;
  }
  return out;
}

//----------------------------------------------------------------------------
/** Decode a run length encoded (RLE) sequence of 8 bit values.
  @param dst  destination position in memory (must be _allocated_!)
  @param src  source location in memory
  @param len  number of bytes to decode
  @param max  number of allocated bytes in destination memory (for range checking)
  @return number of bytes written to destination memory. If max would
          have been exceeded, max+1 is returned
  @see encodeRLEFast
*/
int vvToolshed::decodeRLEFast(uchar* dst, uchar* src, int len, int max)
{
  int count;                                      // RLE counter
  int out=0;                                      // counter for written output bytes

  while (len > 0)
  {
    count = (int)*src++;
    if (count > 127)                              // replicate run?
    {
      count -= 127;                               // remove bias
      if (out+count <= max)                       // don't exceed allocated memory array
        memset(dst, *src++, count);
      else
      {
        if (out < max)                            // write as much as possible
          memset(dst, *src++, max-out);
        return max+1;
      }
      len -= 2;
    }
    else                                          // literal run
    {
      ++count;                                    // remove bias
      if (out+count <= max)                       // don't exceed allocated memory array
        memcpy(dst, src, count);
      else
      {
        if (out < max)                            // write as much as possible
          memcpy(dst, src, max-out);
        return max+1;
      }
      src += count;
      len -= count + 1;
    }
    dst += count;
    out += count;
  }
  return out;
}

//----------------------------------------------------------------------------
/** Get the number of (logical) system CPUs.
  @return number of processors, or -1 if unable to determine it
*/
int vvToolshed::getNumProcessors()
{
#ifdef _WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;
#elif defined(__hpux)
  struct pst_dynamic psd;
  if (pstat_getdynamic(&psd, sizeof(psd), (size_t)1, 0) != -1)
  {
    int nspu = psd.psd_proc_cnt;
    return nspu;
  }
  else
  {
    return 1;
  }
#elif defined(__APPLE__)
  return 1;
#else
  long numProcs;
#ifdef __sgi
  numProcs = sysconf(_SC_NPROC_CONF);
#else
  numProcs = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  if (numProcs < 1) return -1;
  else return numProcs;
#endif
}

//----------------------------------------------------------------------------
/** Returns RGBA texture values for a hue/saturation color chooser.
 Texture values are returned in data as 4 bytes per texel, bottom to top,
 ordered RGBARGBARGBA...
 @param width,height   width and height of texture [pixels]
 @param brightness     brightness of color values [0..1]
 @param data           pointer to allocated memory space providing width * height * 4 bytes
*/
void vvToolshed::makeColorBoardTexture(int width, int height, float brightness,
uchar* data)
{
  float h, s, v;                                  // hue, saturation, value
  float r, g, b;                                  // RGB
  float nx, ny;                                   // x and y normalized to range [0..1]
  float dx, dy;                                   // distance from center
  int   i = 0;                                    // index of current texel element
  int   x, y;                                     // current texel position

  for (y=0; y<height; ++y)
    for (x=0; x<width; ++x)
  {
    nx = (float)x / (float)(width-1);
    ny = (float)y / (float)(height-1);
    dx = 2.0f * nx - 1.0f;
    dy = 2.0f * ny - 1.0f;
    if ( (dx * dx + dy * dy) > 1.0f)
    {
      // Outer area is black:
      data[i++] = (uchar)0;                       // red
      data[i++] = (uchar)0;                       // green
      data[i++] = (uchar)0;                       // blue
      data[i++] = (uchar)255;                     // alpha
    }
    else
    {
      v = brightness;                             // circle area has requested brightness
      convertXY2HS(nx, ny, &h, &s);
      vvToolshed::HSBtoRGB(h, s, v, &r, &g, &b);
      data[i++] = (uchar)(r * 255.0f);            // red
      data[i++] = (uchar)(g * 255.0f);            // green
      data[i++] = (uchar)(b * 255.0f);            // blue
      data[i++] = (uchar)255;                     // alpha
    }
  }
}

//----------------------------------------------------------------------------
/** The given x|y coordinates of the mouse are translated to hue and
    saturation values.
 Mouse coordinate 0|0 is bottom left, 1|1 is top right.
 Hue and saturation values are in range [0..1].
*/
void vvToolshed::convertXY2HS(float x, float y, float* hue, float* saturation)
{
  float dx, dy;                                   // distance from center of x/y area

  // Determine hue:
  dx = x - 0.5f;
  dy = y - 0.5f;

  if (dx==0.0f)
  {
    if (dy>=0.0f) *hue = 0.0f;
    else *hue = 180.0f;
  }
  else if (dy==0.0f)
  {
    if (dx>0.0f) *hue = 90.0f;
    else *hue = 270.0f;
  }
  else
  {
    if      (dx>0.0f && dy>0.0f) *hue = atanf(dx / dy);
    else if (dx>0.0f && dy<0.0f) *hue = TS_PI * 0.5f + atanf(-dy / dx);
    else if (dx<0.0f && dy<0.0f) *hue = TS_PI + atanf(-dx / -dy);
    else                         *hue = TS_PI * 1.5f + atanf(dy / -dx);
  }
  *hue /= (2.0f * TS_PI);
  *hue = ts_clamp(*hue, 0.0f, 1.0f);

  // Determine saturation:
  dx *= 2.0f;
  dy *= 2.0f;
  *saturation = sqrtf(dx * dx + dy * dy);
  *saturation = ts_clamp(*saturation, 0.0f, 1.0f);
}

//----------------------------------------------------------------------------
/** The given hue and saturation values are converted to mouse x|y coordinates.
 Hue and saturation values are in range [0..1].
 Mouse coordinate 0|0 is bottom left, 1|1 is top right.
*/
void vvToolshed::convertHS2XY(float hue, float saturation, float* x, float* y)
{
  float angle;                                    // angle of point xy position within color circle
  float dx, dy;                                   // point position relative to circle midpoint

  angle = hue * 2.0f * TS_PI;
  dx = 0.5f * saturation * sinf(angle);
  dy = 0.5f * saturation * cosf(angle);
  *x = dx + 0.5f;
  *y = dy + 0.5f;
  *x = ts_clamp(*x, 0.0f, 1.0f);
  *y = ts_clamp(*y, 0.0f, 1.0f);
}

//----------------------------------------------------------------------------
/** Read an unsigned char value from a file.
 */
uchar vvToolshed::read8(FILE* src)
{
  uchar val;
  size_t retval;

  retval=fread(&val, 1, 1, src);
  if (retval!=1)
  {
    std::cerr<<"vvToolshed::read8 fread failed"<<std::endl;
    return 0;
  }
  return val;
}

//----------------------------------------------------------------------------
/** Write an unsigned char value to a file.
  @return number of bytes written
*/
int vvToolshed::write8(FILE* dst, uchar val)
{
  size_t retval;
  retval=fwrite(&val, 1, 1, dst);
  if (retval!=1)
  {
    std::cerr<<"vvToolshed::write8 fwrite failed"<<std::endl;
    return 0;
  }
  return 1;
}

//----------------------------------------------------------------------------
/** Read an unsigned short value system independently from a file.
 */
ushort vvToolshed::read16(FILE* src, vvToolshed::EndianType end)
{
  uchar buf[2];
  int val;

  size_t retval;

  retval=fread(buf, 2, 1, src);

  if (retval!=1)
  {
    std::cerr<<"vvToolshed::read16 fread failed"<<std::endl;
    return 0;
  }
  if (end==VV_LITTLE_END)
  {
    val = (int)buf[0] + (int)buf[1] * (int)256;
  }
  else
  {
    val = (int)buf[0] * (int)256 + (int)buf[1];
  }
  return (ushort)val;
}

//----------------------------------------------------------------------------
/** Write an unsigned short value system independently to a file.
  @return number of bytes written
*/
int vvToolshed::write16(FILE* fp, ushort val, vvToolshed::EndianType end)
{
  uchar buf[2];

  if (end==VV_LITTLE_END)
  {
    buf[0] = (uchar)(val & 0xFF);
    buf[1] = (uchar)(val >> 8);
  }
  else
  {
    buf[0] = (uchar)(val >> 8);
    buf[1] = (uchar)(val & 0xFF);
  }
  size_t retval;
  retval=fwrite(buf, 2, 1, fp);
  if (retval!=1)
  {
    std::cerr<<"vvToolshed::write16 fwrite failed"<<std::endl;
    return 0;
  }
  return 2;
}

//----------------------------------------------------------------------------
/** Read an unsigned long value system independently from a file.
 */
uint32_t vvToolshed::read32(FILE* src, vvToolshed::EndianType end)
{
  uchar buf[4];
  uint32_t val;

  size_t retval;
  retval=fread(buf, 4, 1, src);
  if (retval!=1)
  {
    std::cerr<<"vvToolshed::read32 fread failed"<<std::endl;
    return 0;
  }

  if (end==VV_LITTLE_END)
  {
    val = (uint32_t)buf[3] * (uint32_t)16777216 + (uint32_t)buf[2] * (uint32_t)65536 +
      (uint32_t)buf[1] * (uint32_t)256 + (uint32_t)buf[0];
  }
  else
  {
    val = (uint32_t)buf[0] * (uint32_t)16777216 + (uint32_t)buf[1] * (uint32_t)65536 +
      (uint32_t)buf[2] * (uint32_t)256 + (uint32_t)buf[3];
  }
  return (uint32_t)val;
}

//----------------------------------------------------------------------------
/** Write an unsigned long value system independently to a file.
  @return number of bytes written
*/
int vvToolshed::write32(FILE* fp, uint32_t val, vvToolshed::EndianType end)
{
  uchar buf[4];

  if (end==VV_LITTLE_END)
  {
    buf[0] = (uchar)(val & 0xFF);
    buf[1] = (uchar)((val >> 8)  & 0xFF);
    buf[2] = (uchar)((val >> 16) & 0xFF);
    buf[3] = (uchar)(val  >> 24);
  }
  else
  {
    buf[0] = (uchar)(val  >> 24);
    buf[1] = (uchar)((val >> 16) & 0xFF);
    buf[2] = (uchar)((val >> 8)  & 0xFF);
    buf[3] = (uchar)(val & 0xFF);
  }
  size_t retval;
  retval=fwrite(buf, 4, 1, fp);
  if (retval!=1)
  {
    std::cerr<<"vvToolshed::write32 fwrite failed"<<std::endl;
    return 0;
  }
  return 4;
}

//----------------------------------------------------------------------------
/** Read a 32 bit float value system independently from a file.
 */
float vvToolshed::readFloat(FILE* src, vvToolshed::EndianType end)
{
  uchar *buf;
  uchar tmp;
  float val;

  size_t retval;
  retval=fread(&val, 4, 1, src);

  if (retval!=1)
  {
    std::cerr<<"vvToolshed::readFloat fread failed"<<std::endl;
    return 0;
  }

  if (getEndianness() != end)
  {
    // Reverse byte order:
    buf = (uchar*)&val;
    tmp = buf[0]; buf[0] = buf[3]; buf[3] = tmp;
    tmp = buf[1]; buf[1] = buf[2]; buf[2] = tmp;
  }
  return val;
}

//----------------------------------------------------------------------------
/** Write a 32 bit float value system independently to a file.
  @return number of bytes written
*/
int vvToolshed::writeFloat(FILE* fp, float val, vvToolshed::EndianType end)
{
  uchar* buf;
  uchar tmp;

  if (getEndianness() != end)
  {
    // Reverse byte order:
    buf = (uchar*)&val;
    tmp = buf[0]; buf[0] = buf[3]; buf[3] = tmp;
    tmp = buf[1]; buf[1] = buf[2]; buf[2] = tmp;
  }

  size_t retval;
  retval=fwrite(&val, 4, 1, fp);
  if (retval!=1)
  {
    std::cerr<<"vvToolshed::writeFloat fwrite failed"<<std::endl;
    return 0;
  }
  return 4;
}

//----------------------------------------------------------------------------
/** Read an unsigned char value from a buffer.
 */
uchar vvToolshed::read8(uchar* src)
{
  return *src;
}

//----------------------------------------------------------------------------
/** Write an unsigned char value to a buffer.
  @return number of bytes written
*/
int vvToolshed::write8(uchar* src, uchar val)
{
  *src = val;
  return sizeof(uchar);
}

//----------------------------------------------------------------------------
/** Read a little endian unsigned short value system independently from a buffer
  (least significant byte first).
*/
ushort vvToolshed::read16(uchar* src, vvToolshed::EndianType end)
{
  int val;

  if (end==VV_LITTLE_END)
  {
    val = (int)src[0] + (int)src[1] * (int)256;
  }
  else
  {
    val = (int)src[0] * (int)256 + (int)src[1];
  }
  return (ushort)val;
}

//----------------------------------------------------------------------------
/** Write a little endian unsigned short value system independently to a buffer
  (least significant byte first).
  @param buf pointer to 2 bytes of _allocated_ memory
  @return number of bytes written
*/
int vvToolshed::write16(uchar* buf, ushort val, vvToolshed::EndianType end)
{
  if (end==VV_LITTLE_END)
  {
    buf[0] = (uchar)(val & 0xFF);
    buf[1] = (uchar)(val >> 8);
  }
  else
  {
    buf[0] = (uchar)(val >> 8);
    buf[1] = (uchar)(val & 0xFF);
  }
  return sizeof(ushort);
}

//----------------------------------------------------------------------------
/** Read a little endian unsigned long value system independently from a buffer.
 Read four bytes in a row in unix-style (least significant byte first).
*/
uint32_t vvToolshed::read32(uchar* buf, vvToolshed::EndianType end)
{
  uint32_t val;

  if (end==VV_LITTLE_END)
  {
    val = (uint32_t)buf[3] * (uint32_t)16777216 + (uint32_t)buf[2] * (uint32_t)65536 +
      (uint32_t)buf[1] * (uint32_t)256 + (uint32_t)buf[0];
  }
  else
  {
    val = (uint32_t)buf[0] * (uint32_t)16777216 + (uint32_t)buf[1] * (uint32_t)65536 +
      (uint32_t)buf[2] * (uint32_t)256 + (uint32_t)buf[3];
  }
  return (uint32_t)val;
}

//----------------------------------------------------------------------------
/** Write an unsigned long value system independently to a buffer.
  @return number of bytes written
*/
int vvToolshed::write32(uchar* buf, uint32_t val, vvToolshed::EndianType end)
{
  if (end==VV_LITTLE_END)
  {
    buf[0] = (uchar)(val & 0xFF);
    buf[1] = (uchar)((val >> 8)  & 0xFF);
    buf[2] = (uchar)((val >> 16) & 0xFF);
    buf[3] = (uchar)(val  >> 24);
  }
  else
  {
    buf[0] = (uchar)(val  >> 24);
    buf[1] = (uchar)((val >> 16) & 0xFF);
    buf[2] = (uchar)((val >> 8)  & 0xFF);
    buf[3] = (uchar)(val & 0xFF);
  }
  return sizeof(uint32_t);
}

//----------------------------------------------------------------------------
/** Read a 32 bit float value system independently from a buffer.
 */
float vvToolshed::readFloat(uchar* buf, vvToolshed::EndianType end)
{
  float  fval;
  uchar* ptr;
  uchar  tmp;

  assert(sizeof(float)==4);
  memcpy(&fval, buf, 4);
  if (getEndianness() != end)
  {
    // Reverse byte order:
    ptr = (uchar*)&fval;
    tmp = ptr[0]; ptr[0] = ptr[3]; ptr[3] = tmp;
    tmp = ptr[1]; ptr[1] = ptr[2]; ptr[2] = tmp;
  }
  return fval;
}

//----------------------------------------------------------------------------
/** Write a 32 bit float value system independently to a buffer.
  @return number of bytes written
*/
int vvToolshed::writeFloat(uchar* buf, float val, vvToolshed::EndianType end)
{
  uchar tmp;

  assert(sizeof(float)==4);
  memcpy(buf, &val, 4);
  if (getEndianness() != end)
  {
    // Reverse byte order:
    tmp = buf[0]; buf[0] = buf[3]; buf[3] = tmp;
    tmp = buf[1]; buf[1] = buf[2]; buf[2] = tmp;
  }
  return sizeof(float);
}

//----------------------------------------------------------------------------
/** Make a float array system independent:
  convert each four byte float to unix-style (most significant byte first).
  @param numValues number of float values in array
  @param array     pointer to float array (size of array must be 4*numValues!)
*/
void vvToolshed::makeArraySystemIndependent(int numValues, float* array)
{
  uchar* buf;                                     // array pointer in uchar format
  int i;
  uchar tmp;                                      // temporary byte value from float array, needed for swapping

  assert(sizeof(float) == 4);
  if (getEndianness()==VV_BIG_END)  return;       // nothing needs to be done

  buf = (uchar*)array;
  for (i=0; i<numValues; ++i)
  {
    // Reverse byte order:
    tmp = buf[0]; buf[0] = buf[3]; buf[3] = tmp;
    tmp = buf[1]; buf[1] = buf[2]; buf[2] = tmp;
    buf += 4;
  }
}

//----------------------------------------------------------------------------
/** Make a system independent float array system dependent:
  convert each four byte float value back to system style.
  @param numValues number of float values in array
  @param array     pointer to float array (size of array must be 4*numValues!)
*/
void vvToolshed::makeArraySystemDependent(int numValues, float* array)
{
  // Swapping bytes is the same as above, therefore use the same code:
  makeArraySystemIndependent(numValues, array);
}

//----------------------------------------------------------------------------
/** Returns the current system's endianness.
 */
vvToolshed::EndianType vvToolshed::getEndianness()
{
  float one = 1.0f;                               // memory representation of 1.0 on big endian machines: 3F 80 00 00
  uchar* ptr;

  ptr = (uchar*)&one;
  if (*ptr == 0x3f) return VV_BIG_END;
  else
  {
    assert(*ptr == 0);
    return VV_LITTLE_END;
  }
}

//----------------------------------------------------------------------------
/** Suspend process for a specific time. If milliseconds are not available
  on a specific system type, seconds are used (e.g., on Cray systems).
  @param msec suspension time [milliseconds]
*/
void vvToolshed::sleep(int msec)
{
#ifdef _WIN32
  Sleep(msec);
#elif CRAY
  sleep(msec / 1000);
#else
  usleep(msec * 1000);
#endif
}

//----------------------------------------------------------------------------
/** Resize a 2D image by resampling with nearest neighbor interpolation.
  Interleaved pixel format is assumed (RGBRGBRGB for 3 byte per pixel (bpp).
  If the pixel formats of source and destination image differ,
  the following approach is taken:
  - srcBPP==1: value will be replicated for first 3 destination pixels, 4th will be 0xFF
  - srcBPP < dstBPP: pixels are padded with 0xFF
  - srcBPP > dstBPP: rightmost components are dropped
  @param dstData must be _pre-allocated_ by the caller with
                 dstWidth*dstHeight*dstBPP bytes!
*/
void vvToolshed::resample(uchar* srcData, int srcWidth, int srcHeight, int srcBPP,
uchar* dstData, int dstWidth, int dstHeight, int dstBPP)
{
  int x, y, i, xsrc, ysrc, minBPP, dstBytes;
  int srcOffset, dstOffset;

  dstBytes = dstWidth * dstHeight * dstBPP;
                                                  // trivial
  if (srcWidth==dstWidth && srcHeight==dstHeight && srcBPP==dstBPP)
  {
    memcpy(dstData, srcData, dstBytes);
    return;
  }

  minBPP = ts_min(srcBPP, dstBPP);

  // Create a black opaque image as basis to work on:
  memset(dstData, 0, dstBytes);                   // black background
  if (dstBPP==4)                                  // make alpha channel opaque
  {
    for (i=0; i<dstBytes; i+=4)
    {
      *(dstData + i + 3) = 255;
    }
  }
  for (y=0; y<dstHeight; ++y)
  {
    for (x=0; x<dstWidth; ++x)
    {
      xsrc = (int)((float)(x * srcWidth)  / (float)dstWidth);
      ysrc = (int)((float)(y * srcHeight) / (float)dstHeight);
      xsrc = ts_clamp(xsrc, 0, srcWidth-1);
      ysrc = ts_clamp(ysrc, 0, srcHeight-1);
      srcOffset = srcBPP * (xsrc + ysrc * srcWidth);
      dstOffset = dstBPP * (x + y * dstWidth);
      if (srcBPP==1)
      {
        memset(dstData + dstOffset, *(srcData + srcOffset), ts_min(dstBPP, 3));
      }
      else
      {
        memcpy(dstData + dstOffset, srcData + srcOffset, minBPP);
      }
    }
  }
}

//----------------------------------------------------------------------------
/** Blend two images together using maximum intensity projection (MIP).
  Both image must be of same width, height, and color depth.
  The color components of the pixels are interleaved.
  @param srcData pointer to source image
  @param width,height image size [pixels]
  @param bpp bytes per pixel
  @param dstData pointer to destination image
*/
void vvToolshed::blendMIP(uchar* srcData, int width, int height, int bpp, uchar* dstData)
{
  uchar* srcPtr;
  uchar* dstPtr;
  int i;

  srcPtr = srcData;
  dstPtr = dstData;
  for (i=0; i<width*height*bpp; ++i)
  {
    *dstPtr = ts_max(*srcPtr, *dstPtr);
    ++srcPtr;
    ++dstPtr;
  }
}

//----------------------------------------------------------------------------
/** @param path will contain the directory that the OS considers current
    @param maxChars specifies the size of the array path provides space for
*/
void vvToolshed::getCurrentDirectory(char* path, int maxChars)
{
  char* buf = new char[maxChars + 64];
#ifdef _WIN32
  GetCurrentDirectory(maxChars, (LPWSTR)buf);
#else
  getcwd(buf, maxChars);
#endif
  extractDirname(path, buf);
}

//----------------------------------------------------------------------------
/** @param path will contain the directory that the OS should consider
 */
void vvToolshed::setCurrentDirectory(const char* path)
{
#ifdef _WIN32
  SetCurrentDirectory((LPCWSTR)path);
#else
  chdir(path);
#endif
}

//----------------------------------------------------------------------------
/** @param path will contain the directory that the current executable is
                located at
    @param maxChars specifies the size of the array path provides space for
*/
void vvToolshed::getProgramDirectory(char* path, int maxChars)
{
#ifdef _WIN32
  char* buf = new char[maxChars + 64];
  GetModuleFileName(NULL, (LPWCH)buf, maxChars);
  extractDirname(path, buf);
#elif _LINUX64BIT                               // This code unfortunately doesn't work under 32 bit
  struct load_module_desc desc;
  dlget(-2, &desc, sizeof(desc));
  strcpy(path, dlgetname(&desc, sizeof(desc), NULL, NULL, NULL));
#else
  getcwd(path, maxChars);                         // TODO: this is not the correct path if the file was started from somewhere else
#endif
}

//----------------------------------------------------------------------------
/** Decode Base64 encoded text to its original binary format.
    Source: http://www.fourmilab.ch/webtools/base64/
    @param src pointer to ASCII source data
    @param numChars number of ASCII characters in source array
    @param dst pointer to _allocated_ binary destination data.
               At least sizeof(src) / 4 * 3 bytes must be allocated.
    @return true on success, false on error
*/
bool vvToolshed::decodeBase64(const char* src, int numChars, uchar* dst)
{
  uchar dtable[256];
  uchar a[4], b[4], o[3];
  int i, j, c, srcIndex=0, dstIndex=0;

  for(i=0;   i<255;  ++i) dtable[i]= uchar(0x80);
  for(i='A'; i<='I'; ++i) dtable[i]= 0+(i-'A');
  for(i='J'; i<='R'; ++i) dtable[i]= 9+(i-'J');
  for(i='S'; i<='Z'; ++i) dtable[i]= 18+(i-'S');
  for(i='a'; i<='i'; ++i) dtable[i]= 26+(i-'a');
  for(i='j'; i<='r'; ++i) dtable[i]= 35+(i-'j');
  for(i='s'; i<='z'; ++i) dtable[i]= 44+(i-'s');
  for(i='0'; i<='9'; ++i) dtable[i]= 52+(i-'0');
  dtable[int('+')]= 62;
  dtable[int('/')]= 63;
  dtable[int('=')]= 0;

  for(;;)
  {
    // Loop over four characters, in which three bytes are encoded:
    for(i=0; i<4; ++i)
    {
      c = src[srcIndex];
      ++srcIndex;

      if (srcIndex==numChars)                     // end of source array reached?
      {
        if (i>0)                                  // end reached in midst of source quadruple?
        {
          cerr << "vvToolshed::decodeBase64: Input array incomplete." << endl;
          return false;
        }
        else return true;                         // finished
      }
      if(dtable[c] & 0x80)
      {
        cerr << "vvToolshed::decodeBase64: Illegal character in input file." << endl;
        return false;
      }
      a[i] = (uchar)c;
      b[i] = (uchar)dtable[c];
    }
    o[0] = (b[0] << 2) | (b[1] >> 4);
    o[1] = (b[1] << 4) | (b[2] >> 2);
    o[2] = (b[2] << 6) |  b[3];
    i = (a[2] == '=') ? 1 : ((a[3] == '=') ? 2 : 3);

    for (j=0; j<i; ++j) dst[dstIndex+j] = o[j];
    if(i<3) return true;                          // finished if incomplete triple has been written
  }
}

//----------------------------------------------------------------------------
/** Interpolate linearly given a x/y value pair and a slope.
 @param x1, y1  point on straight line (x/y value pair)
 @param slope   slope of line
 @param x       x value for which to find interpolated y value
 @return interpolated y value
*/
float vvToolshed::interpolateLinear(float x1, float y1, float slope, float x)
{
  return (y1 + slope * (x - x1));
}

//----------------------------------------------------------------------------
/** Interpolate linearly between two x/y value pairs.
 @param x1, y1  one x/y value pair
 @param x2, y2  another x/y value pair
 @param x       x value for which to find interpolated y value
 @return interpolated y value
*/
float vvToolshed::interpolateLinear(float x1, float y1, float x2, float y2, float x)
{
  if (x1==x2) return ts_max(y1, y2);              // on equal x values: return maximum value
  if (x1 > x2)                                    // make x1 less than x2
  {
    ts_swap(x1, x2);
    ts_swap(y1, y2);
  }
  return (y2 - y1) * (x - x1) / (x2 - x1) + y1;
}

//----------------------------------------------------------------------------
/** Make a list of files and folders in a path.
  @param path location to search in
  @return fileNames and folderNames, alphabetically sorted
*/
bool vvToolshed::makeFileList(std::string& path, std::list<std::string>& fileNames, std::list<std::string>& folderNames)
{
#ifdef _WIN32
  WIN32_FIND_DATA fileInfo;
  HANDLE fileHandle;
  string searchPath;
  
  searchPath = path + "\\*";
  fileHandle = FindFirstFile((LPCWSTR)searchPath.c_str(), &fileInfo);
  if (fileHandle == INVALID_HANDLE_VALUE) 
  {
    cerr << "FindFirstFile failed: " << GetLastError() << endl;
    return false;
  }
  do  // add all files and directory in specified path to lists
  {
    cerr << "file=" << fileInfo.cFileName << endl;
    if(fileInfo.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY)
    {
      folderNames.push_back((const std::string &)fileInfo.cFileName);
    }
    else
    {
      fileNames.push_back((const std::string &)fileInfo.cFileName);
    }
  }
  while (FindNextFile((HANDLE)fileHandle, &fileInfo));   // was another file found?
  FindClose(fileHandle);
#else
  DIR* dirHandle;
  struct dirent* entry;
  struct stat statbuf;

  dirHandle = opendir(path.c_str());
  if (dirHandle==NULL)
  {
    cerr << "Cannot read directory: " << path << endl;
    return false;
  }
  if (chdir(path.c_str()) != 0)
  {
    const int PATH_SIZE = 256;
    char cwd[PATH_SIZE];
    cerr << "Cannot chdir to " << path <<
      ". Searching for files in " << getcwd(cwd, PATH_SIZE) << endl;
  }
  while ((entry=readdir(dirHandle)) != NULL)
  {
    stat(entry->d_name, &statbuf);
    if (S_ISDIR(statbuf.st_mode))      // found a folder?
    {
      folderNames.push_back(entry->d_name);
    }
    else      // found a file
    {
      fileNames.push_back(entry->d_name);
    }
  }

  fileNames.sort();
  folderNames.sort();

  closedir(dirHandle);
#endif  
  return true;
}

//----------------------------------------------------------------------------
/** Return the next string in the list after a given one
  @param listStrings list of strings to search
  @param knownEntry we're looking for the entry after this one
  @param nextEntry returned string, or "" if nothing found
  @return true if next string found, or false if not
*/
bool vvToolshed::nextListString(list<string>& listStrings, string& knownEntry, string& nextEntry)
{
  list<string>::const_iterator iter;
  for (iter=listStrings.begin(); iter != listStrings.end(); ++iter)
  {
    if ((*iter)==knownEntry) 
    {
      ++iter;
      if (iter == listStrings.end()) return false;  // end of list reached
      else 
      {  
        nextEntry = *iter;
        return true;
      }
    }
  }
  nextEntry = "";
  return false;   // knownEntry has not been found
}

void vvToolshed::quickSort(int* numbers, int arraySize)
{
  qSort(numbers, 0, arraySize - 1);
}

void vvToolshed::qSort(int* numbers, int left, int right)
{
  int pivot, l_hold, r_hold;

  l_hold = left;
  r_hold = right;
  pivot  = numbers[left];
  while (left < right)
  {
    while ((numbers[right] >= pivot) && (left < right))
    {
      --right;
    }
    if (left != right)
    {
      numbers[left] = numbers[right];
      ++left;
    }
    while ((numbers[left] <= pivot) && (left < right))
    {
      ++left;
    }
    if (left != right)
    {
      numbers[right] = numbers[left];
      --right;
    }
  }
  numbers[left] = pivot;
  pivot = left;
  left  = l_hold;
  right = r_hold;
  if (left < pivot)  qSort(numbers, left, pivot-1);
  if (right > pivot) qSort(numbers, pivot+1, right);
}

char* vvToolshed::file2string(const char *filename)
{
	FILE *fp = fopen(filename,"rt");

	if (fp == NULL)
	{
		cerr << "File NOT found: " << filename << endl;
		return NULL;
	}

	fseek(fp, 0, SEEK_END);
	int count = ftell(fp);
	rewind(fp);

	char* content = new char[count+1];

	count = fread(content,sizeof(char),count,fp);
	content[count] = '\0';
	
	fclose(fp);

	return content;
}



//----------------------------------------------------------------------------
/// Main function for standalone test mode.
#ifdef VV_STANDALONE
int main(int, char**)
{
#ifdef _WIN32
  char* pathname={"c:\\user\\testfile.dat"};
#else
  char* pathname={"/usr/local/testfile.dat"};
#endif
  char  teststring[256];

  cout << "ts_max(2,9)  = " << ts_max(2,9)  << endl;
  cout << "ts_min(2,9)  = " << ts_min(2,9)  << endl;
  cout << "ts_abs(-7)   = " << ts_abs(-7)   << endl;
  cout << "ts_sgn(-9.1) = " << ts_sgn(-9.1) << endl;
  cout << "ts_zsgn(0.0) = " << ts_zsgn(0.0) << endl;
  cout << "ts_zsgn(-2)  = " << ts_zsgn(-2)  << endl;
  cout << "ts_clamp(1.2, 1.0, 2.0)  = " << ts_clamp(1.2f, 1.0f, 2.0f)  << endl;
  cout << "ts_clamp(-0.5, 1.0, 2.0)  = " << ts_clamp(0.5f, 1.0f, 2.0f)  << endl;
  cout << "ts_clamp(2.1, 1.0, 2.0)  = " << ts_clamp(2.1f, 1.0f, 2.0f)  << endl;

  cout << "isSuffix(" << pathname << "), 'Dat' = ";
  if (vvToolshed::isSuffix(pathname, "Dat") == true)
    cout << "true" << endl;
  else
    cout << "false" << endl;

  cout << "isSuffix(" << pathname << "), 'data' = ";
  if (vvToolshed::isSuffix(pathname, "data") == true)
    cout << "true" << endl;
  else
    cout << "false" << endl;

  vvToolshed::extractFilename(teststring, pathname);
  cout << "extractFilename(" << pathname << ") = " << teststring << endl;

  vvToolshed::extractDirname(teststring, pathname);
  cout << "extractDirname(" << pathname << ") = " << teststring << endl;

  vvToolshed::extractExtension(teststring, pathname);
  cout << "extractExtension(" << pathname << ") = " << teststring << endl;

  vvToolshed::extractBasename(teststring, pathname);
  cout << "extractBasename(" << pathname << ") = " << teststring << endl;

  cout << "getTextureSize(84) = " << vvToolshed::getTextureSize(84) << endl;

  char* testData = {"ABABACACACABABABACABCD"};
  char encoded[100];
  char decoded[100];
  int len;
  int bpc = 2;
  cout << "Unencoded: " << testData << endl;
  len = vvToolshed::encodeRLE((uchar*)encoded, (uchar*)testData, strlen(testData), bpc, 100);
  len = vvToolshed::decodeRLE((uchar*)decoded, (uchar*)encoded, len, bpc, 100);
  decoded[len] = '\0';
  cout << "Decoded:   " << decoded << endl;

  return 1;
}
#endif

//============================================================================
// End of File
//============================================================================
