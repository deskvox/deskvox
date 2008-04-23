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

#ifndef _VVTOOLSHED_H_
#define _VVTOOLSHED_H_

#include <list>
#include <string>

#include <stdio.h>
#ifndef __sgi
#ifndef WIN32
#include <stdint.h>
#else
typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif
#else
#include <inttypes.h>
#endif

#include "vvexport.h"
#include "vvarray.h"

//============================================================================
// Constant Definitions
//============================================================================

                                                  ///< compiler independent definition for pi
const float TS_PI = 3.1415926535897932384626433832795028841971693993751058f;

//============================================================================
// Type Declarations
//============================================================================

typedef unsigned char   uchar;                    ///< abbreviation for unsigned char
typedef unsigned short  ushort;                   ///< abbreviation for unsigned short
typedef unsigned int    uint;                     ///< abbreviation for unsigned int
typedef unsigned long   ulong;                    ///< abbreviation for unsigned long
typedef signed   char   schar;                    ///< abbreviation for signed char
typedef signed   short  sshort;                   ///< abbreviation for signed short
typedef signed   int    sint;                     ///< abbreviation for signed int
typedef signed   long   slong;                    ///< abbreviation for signed long

//============================================================================
// Function Templates
//============================================================================

/// @return the maximum of two values
template <class C> inline C ts_max(const C a, const C b)
{
  return (a < b) ? b : a;
}

/// @return the maximum of three values
template <class C> inline C ts_max(const C a, const C b, const C c)
{
  return ts_max(a, ts_max(b, c));
}

/// @return the minimum of two values
template <class C> inline C ts_min(const C a, const C b)
{
  return (a > b) ? b : a;
}

/// @return the minimum of three values
template <class C> inline C ts_min(const C a, const C b, const C c)
{
  return ts_min(a, ts_min(b, c));
}

/// @return the absolute value of a value
template <class C> inline C ts_abs(const C a)
{
  return (a < 0) ? (-a) : a;
}

/// @return the sign (-1 or 1) of a value
template <class C> inline C ts_sgn(const C a)
{
  return (a < 0) ? (-1) : 1;
}

/// @return the sign or zero(-1, 0, 1)
template <class C> inline C ts_zsgn(const C a)
{
  return (a < 0) ? (-1) : (a > 0 ? 1 : 0);
}

/// @return the result of value a clamped between left and right
template <class C> inline C ts_clamp(const C a, const C left, const C right)
{
  if (a < left)  return left;
  if (a > right) return right;
  return a;
}

/// Swaps the values a and b
template <class C> inline void ts_swap(C a, C b)
{
  C bak = a;
  a     = b;
  b     = bak;
}

/// @param a a value
/// @return the square of a
template <class C> inline C ts_sqr(C a)
{
  return a * a;
}

//============================================================================
// Class Definitions
//============================================================================

/** Collection of miscellaneous tools.
    Consists of static helper functions which are project independent.
    @author Juergen Schulze-Doebold

    <B>Terminology for extraction functions:</B><BR>
    Example: c: \ test \ vfview.exe
    <UL>
      <LI>Pathname  = c: \ test \ vfview.exe
      <LI>Dirname   = c: \ test \ 
      <LI>Extension = exe
      <LI>Filename  = vfview.exe
<LI>Basename  = vfview
</UL>
*/
class VIRVOEXPORT vvToolshed
{
  private:
    static int progressSteps;                     ///< total number of progress steps

  public:
    enum EndianType                               /// endianness
    {
      VV_LITTLE_END,                              ///< little endian: low-order byte is stored first
      VV_BIG_END                                  ///< big endian: hight-order byte is stored first
    };

    static int     strCompare(const char*, const char*);
    static int     strCompare(const char*, const char*, int n);
    static bool    isSuffix(const char*, const char*);
    static void    HSBtoRGB(float*, float*, float*);
    static void    HSBtoRGB(float, float, float, float*, float*, float*);
    static void    RGBtoHSB(float*, float*, float*);
    static void    RGBtoHSB(float, float, float, float*, float*, float*);
    static void    strcpyTail(char*, const char*, char);
    static string  strcpyTail(const string, char);
    static void    strcpyHead(char*, const char*, char);
    static void    strTrim(char*);
    static void    extractFilename(char*, const char*);
    static string  extractFilename(const string);
    static void    extractDirname(char*, const char*);
    static string  extractDirname(const string);
    static void    extractExtension(char*, const char*);
    static string  extractExtension(const string);
    static void    extractBasename(char*, const char*);
    static string  extractBasename(const string);
    static void    extractBasePath(char*, const char*);
    static void    replaceExtension(char*, const char*, const char*);
    static bool    increaseFilename(char*);
    static bool    increaseFilename(string&);
    static void    draw3DLine(int, int, int, int, int, int, uchar*, uchar*, int, int, int, int);
    static void    draw2DLine(int, int, int, int, uint, uchar*, int, int, int);
    static int     getTextureSize(int);
    static bool    isFile(const char*);
    static bool    isDirectory(const char*);
    static long    getFileSize(const char*);
    static void    getMinMax(const float*, int, float*, float*);
    static void    getMinMax(const uchar*, int, int*, int*);
    static void    getMinMax16bitBE(const uchar*, int, int*, int*);
    static void    getMinMaxAlpha(const uchar*, int, int*, int*);
    static void    getMinMaxIgnore(const float*, int, float, float*, float*);
    static void    convertUChar2Float(const uchar*, float*, int);
    static void    convertFloat2UChar(const float*, uchar*, int);
    static void    convertFloat2UCharClamp(const float*, uchar*, int, float, float);
    static void    convertFloat2ShortClamp(const float*, uchar*, int, float, float);
    static void    convertFloat2UCharClampZero(const float*, uchar*, int, float, float, float);
    static int     getLargestPrimeFactor(int);
    static int     round(float);
    static void    initProgress(int);
    static void    printProgress(int);
    static int     encodeRLE(uchar*, uchar*, int, int, int);
    static int     decodeRLE(uchar*, uchar*, int, int, int);
    static int     encodeRLEFast(uchar*, uchar*, int, int);
    static int     decodeRLEFast(uchar*, uchar*, int, int);
    static int     getNumProcessors();
    static void    makeColorBoardTexture(int, int, float, uchar*);
    static void    convertXY2HS(float, float, float*, float*);
    static void    convertHS2XY(float, float, float*, float*);
    static uchar   read8(FILE*);
    static int     write8(FILE*, uchar);
    static ushort  read16(FILE*, vvToolshed::EndianType = VV_BIG_END);
    static int     write16(FILE*, ushort, vvToolshed::EndianType = VV_BIG_END);
    static uint32_t read32(FILE*, vvToolshed::EndianType = VV_BIG_END);
    static int     write32(FILE*, uint32_t, vvToolshed::EndianType = VV_BIG_END);
    static float   readFloat(FILE*, vvToolshed::EndianType = VV_BIG_END);
    static int     writeFloat(FILE*, float, vvToolshed::EndianType = VV_BIG_END);
    static uchar   read8(uchar*);
    static int     write8(uchar*, uchar);
    static ushort  read16(uchar*, vvToolshed::EndianType = VV_BIG_END);
    static int     write16(uchar*, ushort, vvToolshed::EndianType = VV_BIG_END);
    static uint32_t read32(uchar*, vvToolshed::EndianType = VV_BIG_END);
    static int     write32(uchar*, uint32_t, vvToolshed::EndianType = VV_BIG_END);
    static float   readFloat(uchar*, vvToolshed::EndianType = VV_BIG_END);
    static int     writeFloat(uchar*, float, vvToolshed::EndianType = VV_BIG_END);
    static void    makeArraySystemIndependent(int, float*);
    static void    makeArraySystemDependent(int, float*);
    static EndianType getEndianness();
    static void    sleep(int);
    static void    resample(uchar*, int, int, int, uchar*, int, int, int);
    static void    blendMIP(uchar*, int, int, int, uchar*);
    static void    getCurrentDirectory(char*, int);
    static void    setCurrentDirectory(const char*);
    static void    getProgramDirectory(char*, int);
    static bool    decodeBase64(const char*, int, uchar*);
    static float   interpolateLinear(float, float, float, float);
    static float   interpolateLinear(float, float, float, float, float);
    static bool    makeFileList(std::string&, std::list<std::string>&, std::list<std::string>&);
    static bool    nextListString(std::list<std::string>&, std::string&, std::string&);
    static void    quickSort(int*, int);
    static void    qSort(int*, int, int); 
    static char*   file2string(const char* filename);
};
#endif

//============================================================================
// End of File
//============================================================================
