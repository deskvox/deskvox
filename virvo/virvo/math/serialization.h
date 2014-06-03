#ifndef VV_MATH_SERIALIZATION_H
#define VV_MATH_SERIALIZATION_H


#include "math.h"


namespace boost
{


namespace serialization
{


template < typename A, size_t D, typename T >
inline void serialize(A& a, virvo::math::vector< D, T >& v, unsigned /* version */ )
{
    for (size_t d = 0; d < D; ++d)
    {
        a & v[d];
    }
}


template < typename A, typename T >
inline void serialize(A& a, virvo::math::vector< 2, T >& v, unsigned /* version */ )
{
    a & v.x;
    a & v.y;
}


template < typename A, typename T >
inline void serialize(A& a, virvo::math::vector< 3, T >& v, unsigned /* version */ )
{
    a & v.x;
    a & v.y;
    a & v.z;
}


template < typename A, typename T >
inline void serialize(A& a, virvo::math::vector< 4, T >& v, unsigned /* version */ )
{
    a & v.x;
    a & v.y;
    a & v.z;
    a & v.w;
}


} // serialization


} // boost


#endif // VV_MATH_SERIALIZATION_H


