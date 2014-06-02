#ifndef VV_MATH_VECTOR_H
#define VV_MATH_VECTOR_H

#include "detail/math.h"

#include "forward.h"

#include <cstddef>


namespace virvo
{


namespace math
{

//--------------------------------------------------------------------------------------------------
// vector2
//

template < typename T >
class vector< 2, T >
{
public:

    typedef T value_type;

    T x;
    T y;

    vector();
    vector(T x, T y);

    explicit vector(T s);
    explicit vector(T const data[2]);

    template < typename U >
    vector(vector< 2, U > const& rhs);

    template < typename U >
    vector& operator=(vector< 2, U > const& rhs);

    T* data();
    T const* data() const;

    T& operator[](size_t i);
    const T& operator[](size_t i) const;

};


//--------------------------------------------------------------------------------------------------
// vector3
//

template < typename T >
class vector< 3, T >
{
public:

    typedef T value_type;

    T x;
    T y;
    T z;

    vector();
    vector(T x, T y, T z);

    explicit vector(T s);

    template < typename U >
    vector(vector< 3, U > const& rhs);

    template < typename U >
    vector& operator=(vector< 3, U > const& rhs);

    T* data();
    T const* data() const;

    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    vector< 2, T >& xy();
    vector< 2, T > const& xy() const;

};


//--------------------------------------------------------------------------------------------------
// vector4
//

template < typename T >
class vector< 4, T >
{
public:

    typedef T value_type;

    T x;
    T y;
    T z;
    T w;

    vector();
    vector(T x, T y, T z, T w);

    explicit vector(T s);
    explicit vector(T const data[4]);

    template < typename U >
    vector(vector< 4, U > const& rhs);

    template < typename U >
    vector& operator=(vector< 4, U > const& rhs);

    T* data();
    T const* data() const;

    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    vector< 3, T >& xyz();
    vector< 3, T > const& xyz() const;

};


} // math


} // virvo


#include "detail/vector.inl"
#include "detail/vector2.inl"
#include "detail/vector3.inl"
#include "detail/vector4.inl"


#endif // VV_MATH_VECTOR_H


