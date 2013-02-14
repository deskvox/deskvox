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


#ifndef VAR_VARIADIC_H_INCLUDED
#define VAR_VARIADIC_H_INCLUDED


#include "vvcompiler.h"


#ifdef CXX_NO_VARIADIC_TEMPLATES


// Call FUNC(TEM_LIST, LIST, APPEND_LIST, N)

// For 0-X args
#define VAR_EXPAND_0X(FUNC) \
    VAR_EXPAND_0(FUNC) \
    VAR_EXPAND_1X(FUNC)

// For 1-X args
#define VAR_EXPAND_1X(FUNC) \
    VAR_EXPAND_1(FUNC) \
    VAR_EXPAND_2X(FUNC)

// For 2-X args
#define VAR_EXPAND_2X(FUNC) \
    VAR_EXPAND_2(FUNC) \
    VAR_EXPAND_3(FUNC) \
    VAR_EXPAND_4(FUNC) \
    VAR_EXPAND_5(FUNC) \
    VAR_EXPAND_6(FUNC) \
    VAR_EXPAND_7(FUNC) \
    VAR_EXPAND_8(FUNC) \
    VAR_EXPAND_9(FUNC)

// Call FUNC(TEM_LIST, LIST, APPEND_LIST, N)

#define VAR_EXPAND_0(FUNC) FUNC(VAR_TEM_LIST0, VAR_LIST0, VAR_APPEND_LIST0, 0)
#define VAR_EXPAND_1(FUNC) FUNC(VAR_TEM_LIST1, VAR_LIST1, VAR_APPEND_LIST1, 1)
#define VAR_EXPAND_2(FUNC) FUNC(VAR_TEM_LIST2, VAR_LIST2, VAR_APPEND_LIST2, 2)
#define VAR_EXPAND_3(FUNC) FUNC(VAR_TEM_LIST3, VAR_LIST3, VAR_APPEND_LIST3, 3)
#define VAR_EXPAND_4(FUNC) FUNC(VAR_TEM_LIST4, VAR_LIST4, VAR_APPEND_LIST4, 4)
#define VAR_EXPAND_5(FUNC) FUNC(VAR_TEM_LIST5, VAR_LIST5, VAR_APPEND_LIST5, 5)
#define VAR_EXPAND_6(FUNC) FUNC(VAR_TEM_LIST6, VAR_LIST6, VAR_APPEND_LIST6, 6)
#define VAR_EXPAND_7(FUNC) FUNC(VAR_TEM_LIST7, VAR_LIST7, VAR_APPEND_LIST7, 7)
#define VAR_EXPAND_8(FUNC) FUNC(VAR_TEM_LIST8, VAR_LIST8, VAR_APPEND_LIST8, 8)
#define VAR_EXPAND_9(FUNC) FUNC(VAR_TEM_LIST9, VAR_LIST9, VAR_APPEND_LIST9, 9)

// Template lists for functions with no leading parameter

#define VAR_TEM_LIST0(MAP)
#define VAR_TEM_LIST1(MAP) \
    template<MAP(1)>
#define VAR_TEM_LIST2(MAP) \
    template<MAP(1), MAP(2)>
#define VAR_TEM_LIST3(MAP) \
    template<MAP(1), MAP(2), MAP(3)>
#define VAR_TEM_LIST4(MAP) \
    template<MAP(1), MAP(2), MAP(3), MAP(4)>
#define VAR_TEM_LIST5(MAP) \
    template<MAP(1), MAP(2), MAP(3), MAP(4), MAP(5)>
#define VAR_TEM_LIST6(MAP) \
    template<MAP(1), MAP(2), MAP(3), MAP(4), MAP(5), MAP(6)>
#define VAR_TEM_LIST7(MAP) \
    template<MAP(1), MAP(2), MAP(3), MAP(4), MAP(5), MAP(6), MAP(7)>
#define VAR_TEM_LIST8(MAP) \
    template<MAP(1), MAP(2), MAP(3), MAP(4), MAP(5), MAP(6), MAP(7), MAP(8)>
#define VAR_TEM_LIST9(MAP) \
    template<MAP(1), MAP(2), MAP(3), MAP(4), MAP(5), MAP(6), MAP(7), MAP(8), MAP(9)>

// Plain lists

#define VAR_LIST0(SEP, MAP)
#define VAR_LIST1(SEP, MAP) \
    MAP(1)
#define VAR_LIST2(SEP, MAP) \
    MAP(1) SEP(2) MAP(2)
#define VAR_LIST3(SEP, MAP) \
    MAP(1) SEP(2) MAP(2) SEP(3) MAP(3)
#define VAR_LIST4(SEP, MAP) \
    MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4)
#define VAR_LIST5(SEP, MAP) \
    MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5)
#define VAR_LIST6(SEP, MAP) \
    MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5) SEP(6) MAP(6)
#define VAR_LIST7(SEP, MAP) \
    MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5) SEP(6) MAP(6) SEP(7) MAP(7)
#define VAR_LIST8(SEP, MAP) \
    MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5) SEP(6) MAP(6) SEP(7) MAP(7) SEP(8) MAP(8)
#define VAR_LIST9(SEP, MAP) \
    MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5) SEP(6) MAP(6) SEP(7) MAP(7) SEP(8) MAP(8) SEP(9) MAP(9)

#define VAR_APPEND_LIST0(SEP, MAP)
#define VAR_APPEND_LIST1(SEP, MAP) \
    SEP(1) MAP(1)
#define VAR_APPEND_LIST2(SEP, MAP) \
    SEP(1) MAP(1) SEP(2) MAP(2)
#define VAR_APPEND_LIST3(SEP, MAP) \
    SEP(1) MAP(1) SEP(2) MAP(2) SEP(3) MAP(3)
#define VAR_APPEND_LIST4(SEP, MAP) \
    SEP(1) MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4)
#define VAR_APPEND_LIST5(SEP, MAP) \
    SEP(1) MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5)
#define VAR_APPEND_LIST6(SEP, MAP) \
    SEP(1) MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5) SEP(6) MAP(6)
#define VAR_APPEND_LIST7(SEP, MAP) \
    SEP(1) MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5) SEP(6) MAP(6) SEP(7) MAP(7)
#define VAR_APPEND_LIST8(SEP, MAP) \
    SEP(1) MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5) SEP(6) MAP(6) SEP(7) MAP(7) SEP(8) MAP(8)
#define VAR_APPEND_LIST9(SEP, MAP) \
    SEP(1) MAP(1) SEP(2) MAP(2) SEP(3) MAP(3) SEP(4) MAP(4) SEP(5) MAP(5) SEP(6) MAP(6) SEP(7) MAP(7) SEP(8) MAP(8) SEP(9) MAP(9)

// List separators

#define VAR_COMMA(_) ,
#define VAR_SEMICOLON(_) ;
#define VAR_OR(_) ||
#define VAR_AND(_) &&

// MAP functions

    // A
#define VAR_TYPE(NUM) \
    A##NUM

    // A&
#define VAR_TYPE_REF(NUM) \
    VAR_TYPE(NUM)&

    // A const&
#define VAR_TYPE_CONST_REF(NUM) \
    VAR_TYPE(NUM) const&

    // A&&
#define VAR_TYPE_REFREF(NUM) \
    VAR_TYPE(NUM)&&

    // class A
#define VAR_CLASS_TYPE(NUM) \
    class VAR_TYPE(NUM)

    // Arg
#define VAR_ARG(NUM) \
    Arg##NUM

    // A Arg
#define VAR_TYPE_ARG(NUM) \
    VAR_TYPE(NUM) VAR_ARG(NUM)

    // A& Arg
#define VAR_TYPE_REF_ARG(NUM) \
    VAR_TYPE_REF(NUM) VAR_ARG(NUM)

    // A const& Arg
#define VAR_TYPE_CONST_REF_ARG(NUM) \
    VAR_TYPE_CONST_REF(NUM) VAR_ARG(NUM)

    // A&& Arg
#define VAR_TYPE_REFREF_ARG(NUM) \
    VAR_TYPE_REFREF(NUM) VAR_ARG(NUM)

    // std::forward<A>(Arg)
#define VAR_FORWARD_ARG(NUM) \
    std::forward<VAR_TYPE(NUM)>(VAR_ARG(NUM))

    // std::move(Arg)
#define VAR_MOVE_ARG(NUM) \
    std::move(VAR_ARG(NUM))


#endif // CXX_NO_VARIADIC_TEMPLATES


#else
#undef VAR_VARIADIC_H_INCLUDED


#ifdef CXX_NO_VARIADIC_TEMPLATES


#undef VAR_EXPAND_0X

#undef VAR_EXPAND_1X

#undef VAR_EXPAND_2X

#undef VAR_EXPAND_0
#undef VAR_EXPAND_1
#undef VAR_EXPAND_2
#undef VAR_EXPAND_3
#undef VAR_EXPAND_4
#undef VAR_EXPAND_5
#undef VAR_EXPAND_6
#undef VAR_EXPAND_7
#undef VAR_EXPAND_8
#undef VAR_EXPAND_9

#undef VAR_TEM_LIST0
#undef VAR_TEM_LIST1
#undef VAR_TEM_LIST2
#undef VAR_TEM_LIST3
#undef VAR_TEM_LIST4
#undef VAR_TEM_LIST5
#undef VAR_TEM_LIST6
#undef VAR_TEM_LIST7
#undef VAR_TEM_LIST8
#undef VAR_TEM_LIST9

#undef VAR_LIST0
#undef VAR_LIST1
#undef VAR_LIST2
#undef VAR_LIST3
#undef VAR_LIST4
#undef VAR_LIST5
#undef VAR_LIST6
#undef VAR_LIST7
#undef VAR_LIST8
#undef VAR_LIST9

#undef VAR_COMMA
#undef VAR_SEMICOLON
#undef VAR_OR
#undef VAR_AND

#undef VAR_TYPE

#undef VAR_TYPE_REF

#undef VAR_TYPE_CONST_REF

#undef VAR_TYPE_REFREF

#undef VAR_CLASS_TYPE

#undef VAR_ARG

#undef VAR_TYPE_ARG

#undef VAR_TYPE_REF_ARG

#undef VAR_TYPE_CONST_REF_ARG

#undef VAR_TYPE_REFREF_ARG

#undef VAR_FORWARD_ARG

#undef VAR_MOVE_ARG


#endif // CXX_NO_VARIADIC_TEMPLATES


#endif // VAR_VARIADIC_H_INCLUDED
