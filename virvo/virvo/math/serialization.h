#ifndef VV_MATH_SERIALIZATION_H
#define VV_MATH_SERIALIZATION_H


#include "math.h"


namespace boost
{


namespace serialization
{


template < typename A, size_t D, typename T >
inline void serialize(A& a, virvo::vector< D, T >& v, unsigned /* version */ )
{
    for (size_t d = 0; d < D; ++d)
    {
        a & v[d];
    }
}


template < typename A, typename T >
inline void serialize(A& a, virvo::vector< 2, T >& v, unsigned /* version */ )
{
    a & v.x;
    a & v.y;
}


template < typename A, typename T >
inline void serialize(A& a, virvo::vector< 3, T >& v, unsigned /* version */ )
{
    a & v.x;
    a & v.y;
    a & v.z;
}


template < typename A, typename T >
inline void serialize(A& a, virvo::vector< 4, T >& v, unsigned /* version */ )
{
    a & v.x;
    a & v.y;
    a & v.z;
    a & v.w;
}


template < typename A, typename T >
inline void serialize(A& a, virvo::matrix< 4, 4, T >& m, unsigned /* version */ )
{
    a & m.col0;
    a & m.col1;
    a & m.col2;
    a & m.col3;
}


template < typename A, typename T >
inline void serialize(A& a, virvo::basic_aabb< T >& b, unsigned /* version */ )
{
    a & b.min;
    a & b.max;
}


template < typename A, typename T >
inline void serialize(A& a, virvo::rectangle< virvo::xywh_layout, T >& r, unsigned /* version */ )
{
    a & r.x;
    a & r.y;
    a & r.w;
    a & r.h;
}


} // serialization


} // boost


#endif // VV_MATH_SERIALIZATION_H


