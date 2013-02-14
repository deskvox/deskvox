// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2012 University of Stuttgart, 2004-2005 Brown University
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


#ifndef CXX_COMPILER_H_INCLUDED
#define CXX_COMPILER_H_INCLUDED


//
// TODO:
//
// merge with vvvariadic, so the define's here can be undefined
// if they're no longer used...
//


//
// Determine compiler
//

#if defined(__clang__)
#define CXX_CLANG    (100 * __clang_major__ + __clang_minor__)
#elif defined(__GNUC__)
#define CXX_GCC      (100 * __GNUC__ + __GNUC_MINOR__)
#elif defined(_MSC_VER)
#define CXX_MSVC     (_MSC_VER)
#endif

#ifndef CXX_CLANG
#define CXX_CLANG 0
#endif
#ifndef CXX_GCC
#define CXX_GCC 0
#endif
#ifndef CXX_MSVC
#define CXX_MSVC 0
#endif


//
// Determine available C++11 features
//

#define CXX_NO_ALIAS_TEMPLATES
#define CXX_NO_AUTO
#define CXX_NO_CONSTEXPR
#define CXX_NO_DECLTYPE
#define CXX_NO_EXPLICIT_CONVERSIONS
#define CXX_NO_GENERALIZED_INITIALIZERS
#define CXX_NO_LAMBDAS
#define CXX_NO_NULLPTR
#define CXX_NO_OVERRIDE_CONTROL
#define CXX_NO_RANGE_FOR
#define CXX_NO_RVALUE_REFERENCES
#define CXX_NO_STATIC_ASSERT
#define CXX_NO_VARIADIC_TEMPLATES

#if CXX_CLANG
#   if __has_feature(cxx_alias_templates)
#       undef CXX_NO_ALIAS_TEMPLATES
#   endif
#   if __has_feature(cxx_auto_type)
#       undef CXX_NO_AUTO
#   endif
#   if __has_feature(cxx_constexpr)
#       undef CXX_NO_CONSTEXPR
#   endif
#   if __has_feature(cxx_decltype)
#       undef CXX_NO_DECLTYPE
#   endif
#   if __has_feature(cxx_explicit_conversions)
#       undef CXX_NO_EXPLICIT_CONVERSIONS
#   endif
#   if __has_feature(cxx_generalized_initializers)
#       undef CXX_NO_GENERALIZED_INITIALIZERS
#   endif
#   if __has_feature(cxx_lambdas)
#       undef CXX_NO_LAMBDAS
#   endif
#   if __has_feature(cxx_nullptr)
#       undef CXX_NO_NULLPTR
#   endif
#   if __has_feature(cxx_override_control)
#       undef CXX_NO_OVERRIDE_CONTROL
#   endif
#   if __has_feature(cxx_range_for)
#       undef CXX_NO_RANGE_FOR
#   endif
#   if __has_feature(cxx_rvalue_references)
#       undef CXX_NO_RVALUE_REFERENCES
#   endif
#   if __has_feature(cxx_static_assert)
#       undef CXX_NO_STATIC_ASSERT
#   endif
#   if __has_feature(cxx_variadic_templates)
#       undef CXX_NO_VARIADIC_TEMPLATES
#   endif
#elif CXX_GCC
#   ifdef __GXX_EXPERIMENTAL_CXX0X__
#       if CXX_GCC >= 403
#           undef CXX_NO_DECLTYPE
#           undef CXX_NO_RVALUE_REFERENCES
#           undef CXX_NO_STATIC_ASSERT
#           undef CXX_NO_VARIADIC_TEMPLATES
#       endif
#       if CXX_GCC >= 404
#           undef CXX_NO_AUTO
#           undef CXX_NO_GENERALIZED_INITIALIZERS
#       endif
#       if CXX_GCC >= 405
#           undef CXX_NO_EXPLICIT_CONVERSIONS
#           undef CXX_NO_LAMBDAS
#       endif
#       if CXX_GCC >= 406
#           undef CXX_NO_CONSTEXPR
#           undef CXX_NO_NULLPTR
#           undef CXX_NO_RANGE_FOR
#       endif
#       if CXX_GCC >= 407
#           undef CXX_NO_ALIAS_TEMPLATES
#           undef CXX_NO_OVERRIDE_CONTROL
#       endif
#   endif
#elif CXX_MSVC
#   if CXX_MSVC >= 1600 // Visual C++ 10.0 (2010)
#       undef CXX_NO_AUTO
#       undef CXX_NO_DECLTYPE
#       undef CXX_NO_LAMBDAS
#       undef CXX_NO_NULLPTR
#       undef CXX_NO_RVALUE_REFERENCES
#       undef CXX_NO_STATIC_ASSERT
#   endif
#   if CXX_MSVC >= 1700 // Visual C++ 11.0 (2012)
#       undef CXX_NO_OVERRIDE_CONTROL
#       undef CXX_NO_RANGE_FOR
#   endif
#   if _MSC_FULL_VER == 170051025 // Visual C++ 12.0 November CTP
#       undef CXX_NO_EXPLICIT_CONVERSIONS
#if 0 // no <initializer_list>...
#       undef CXX_NO_GENERALIZED_INITIALIZERS
#endif
#       undef CXX_NO_VARIADIC_TEMPLATES
#   endif
#endif


#ifndef CXX_NO_OVERRIDE_CONTROL
#define CXX_OVERRIDE override
#define CXX_FINAL final
#else
#define CXX_OVERRIDE
#define CXX_FINAL
#endif


//
// Macros to work with compiler warnings/errors
//

// Use like: CXX_MSVC_WARNING_DISABLE(4996)
#if CXX_MSVC
#   define CXX_MSVC_WARNING_SUPPRESS(X) \
        __pragma(warning(suppress : X))
#   define CXX_MSVC_WARNING_PUSH_LEVEL(X) \
        __pragma(warning(push, X))
#   define CXX_MSVC_WARNING_PUSH \
        __pragma(warning(push))
#   define CXX_MSVC_WARNING_POP \
        __pragma(warning(pop))
#   define CXX_MSVC_WARNING_DEFAULT(X) \
        __pragma(warning(default : X))
#   define CXX_MSVC_WARNING_DISABLE(X) \
        __pragma(warning(disable : X))
#   define CXX_MSVC_WARNING_ERROR(X) \
        __pragma(warning(error : X))
#   define CXX_MSVC_WARNING_PUSH_DISABLE(X) \
        __pragma(warning(push)) \
        __pragma(warning(disable : X))
#else
#   define CXX_MSVC_WARNING_SUPPRESS(X)
#   define CXX_MSVC_WARNING_PUSH_LEVEL(X)
#   define CXX_MSVC_WARNING_PUSH
#   define CXX_MSVC_WARNING_POP
#   define CXX_MSVC_WARNING_DEFAULT(X)
#   define CXX_MSVC_WARNING_DISABLE(X)
#   define CXX_MSVC_WARNING_ERROR(X)
#   define CXX_MSVC_WARNING_PUSH_DISABLE(X)
#endif


// Use like: CXX_GCC_DIAGNOSTIC_IGNORE("-Wuninitialized")
#if CXX_CLANG || CXX_GCC
#   define CXX_GCC_DIAGNOSTIC_PUSH \
        _Pragma("GCC diagnostic push")
#   define CXX_GCC_DIAGNOSTIC_POP \
        _Pragma("GCC diagnostic pop")
#   define CXX_GCC_DIAGNOSTIC_IGNORE(X) \
        _Pragma("GCC diagnostic ignored \"" X "\"")
#   define CXX_GCC_DIAGNOSTIC_WARNING(X) \
        _Pragma("GCC diagnostic warning \"" X "\"")
#   define CXX_GCC_DIAGNOSTIC_ERROR(X) \
        _Pragma("GCC diagnostic error \"" X "\"")
#else
#   define CXX_GCC_DIAGNOSTIC_PUSH
#   define CXX_GCC_DIAGNOSTIC_POP
#   define CXX_GCC_DIAGNOSTIC_IGNORE(X)
#   define CXX_GCC_DIAGNOSTIC_WARNING(X)
#   define CXX_GCC_DIAGNOSTIC_ERROR(X)
#endif


#endif // CXX_COMPILER_H_INCLUDED
