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

#ifndef _VVVIRVO_H_
#define _VVVIRVO_H_

/** \mainpage Virvo
  <DL>
    <DT><B>Functionality</B>         <DD>VIRVO stands for VIrtual Reality VOlume renderer.
                                     It is a library for real-time volume rendering with hardware accelerated texture mapping.
    <DT><B>Developer Information</B> <DD>The library does not depend on other libraries except OpenGL.
                                     If you want to use Nvidia Cg pixel shaders,
                                     you need to define HAVE_CG at compile time.
                                     The main rendering classes are vvRenderer and vvTexRend. You can
                                     create new rendering classes by deriving them from vvRenderer.
                                     Transfer functions for volume rendering are managed by vvTransFunc.
                                     The class vvSocket allows system independent socket communication,
not limited to the transfer of volume data. The classes
vvArray and vvSLList/vvSLNode allow using STL-style vectors and lists
without actually linking with STL (which is problematic if the code
is supposed to be system independent). vvDicom is a pretty good DICOM
image file reader which can be extended to any unknown formats. vvVector3/4
and vvMatrix are components of vvVecmath, a useful library for linear algebra.
<DT><B>Copyright</B>             <DD>(c) 1999-2005 J&uuml;rgen P. Schulze. All rights reserved.
<DT><B>Email</B>                 <DD>jschulze@ucsd.edu
<DT><B>Institution</B>           <DD>Brown University
</DL>
*/

#ifdef _WIN32 // actually should check for visual c++
#pragma warning(disable: 4514)                    // disable warning about unreferenced inline function
#endif

// Current version (to be updated on every release):
#define VV_VERSION "2"                            // major version change
#define VV_RELEASE "01b"                          // release counter
#define VV_YEAR 2005                              // year of release
#endif

//============================================================================
// End of File
//============================================================================
