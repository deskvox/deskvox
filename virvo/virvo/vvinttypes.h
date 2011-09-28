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

#ifndef _VVINTTYPES_H_
#define _VVINTTYPES_H_

//============================================================================
// Type Declarations
//============================================================================

#ifdef _WIN32
/* code copied from Python's pyconfig.h,
 * to avoid different definition of  ssize_t */
#ifdef _WIN64
typedef __int64 ssize_t;
#else
#if !defined(_WIN32_WCE) && !defined(__MINGW32__) 
typedef _W64 int ssize_t;
#endif
#endif
/* end copy */
#endif

#ifndef __sgi
#ifndef _MSC_VER
#include <stdint.h>
#else
#ifdef HAVE_GDCM
#include "stdint.h"
#else
#if (_MSC_VER >= 1600)  /* VisualStudio 2010 comes with stdint.h */
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
#endif
#endif
#else
#include <inttypes.h>
#endif

typedef unsigned char   uchar;                    ///< abbreviation for unsigned char
typedef unsigned short  ushort;                   ///< abbreviation for unsigned short
typedef unsigned int    uint;                     ///< abbreviation for unsigned int
typedef unsigned long   ulong;                    ///< abbreviation for unsigned long
typedef signed   char   schar;                    ///< abbreviation for signed char
typedef signed   short  sshort;                   ///< abbreviation for signed short
typedef signed   int    sint;                     ///< abbreviation for signed int
typedef signed   long   slong;                    ///< abbreviation for signed long

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
