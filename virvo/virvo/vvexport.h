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

#ifndef VV_EXPORT_H
#define VV_EXPORT_H

#ifdef _MSC_VER
# define VV_DLLEXPORT __declspec(dllexport)
# define VV_DLLIMPORT __declspec(dllimport)
# define VV_DLLHIDDEN
#elif defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 4)
# define VV_DLLEXPORT __attribute__((visibility("default")))
# define VV_DLLIMPORT __attribute__((visibility("default")))
# define VV_DLLHIDDEN __attribute__((visibility("hidden")))
#else
# define VV_DLLEXPORT
# define VV_DLLIMPORT
# define VV_DLLHIDDEN
#endif

#ifndef VIRVO_STATIC
# ifdef VIRVO_EXPORTS
#   define VVAPI VV_DLLEXPORT
# else
#   define VVAPI VV_DLLIMPORT
# endif
# define VVLOCAL VV_DLLHIDDEN
#else
# define VVAPI
# define VVLOCAL
#endif

// compatibility...
#define VIRVOEXPORT VVAPI

#endif
