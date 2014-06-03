#ifndef VV_SIMD_MATRIX_H
#define VV_SIMD_MATRIX_H


#include "vec.h"

#include <ostream>

namespace virvo
{

namespace math
{


template < >
class matrix< 4, 4, simd::float4 >
{
public:

    typedef simd::float4 column_type;

    column_type col0;
    column_type col1;
    column_type col2;
    column_type col3;


    matrix();
    explicit matrix(float const data[16]);

    column_type& operator()(size_t col);
    column_type const& operator()(size_t col) const;

};


} // math

} // virvo


#include "../detail/simd/matrix4.inl"


#endif // VV_SIMD_MATRIX_H


