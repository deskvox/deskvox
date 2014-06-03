#include "../../vector.h"


namespace virvo
{


namespace math
{


//--------------------------------------------------------------------------------------------------
// matrix4 members
//

VV_FORCE_INLINE matrix< 4, 4, simd::float4 >::matrix()
{
}

VV_FORCE_INLINE matrix< 4, 4, simd::float4 >::matrix(float const data[16])
    : col0(&data[ 0])
    , col1(&data[ 4])
    , col2(&data[ 8])
    , col3(&data[12])
{
}

VV_FORCE_INLINE simd::float4& matrix< 4, 4, simd::float4 >::operator()(size_t col)
{
    return *(reinterpret_cast< simd::float4* >(this) + col);
}

VV_FORCE_INLINE simd::float4 const& matrix< 4, 4, simd::float4 >::operator()(size_t col) const
{
    return *(reinterpret_cast< simd::float4 const* >(this) + col);
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

VV_FORCE_INLINE matrix< 4, 4, simd::float4 > operator*(
    matrix< 4, 4, simd::float4 > const& a, matrix< 4, 4, simd::float4 > const& b)
{

    using simd::float4;
    using simd::shuffle;

    matrix< 4, 4, simd::float4 > result;
    for (size_t i = 0; i < 4; ++i)
    {
        float4 col = shuffle< 0, 0, 0, 0 >(a(i)) * b(0);
        col += shuffle< 1, 1, 1, 1 >(a(i)) * b(1);
        col += shuffle< 2, 2, 2, 2 >(a(i)) * b(2);
        col += shuffle< 3, 3, 3, 3 >(a(i)) * b(3);
        result(i) = col;
    }

    return result;

}

VV_FORCE_INLINE vector< 4, simd::float4 > operator*(
    matrix< 4, 4, simd::float4 > const& m, vector< 4, simd::float4 > const& v)
{

    matrix< 4, 4, simd::float4 > tmp;

    tmp.col0 = v.x;
    tmp.col1 = v.y;
    tmp.col2 = v.z;
    tmp.col3 = v.w;

    matrix< 4, 4, simd::float4 > res = m * tmp;
    return vector< 4, simd::float4 >( res.col0, res.col1, res.col2, res.col3 );

}


} // math


} // virvo


