namespace virvo
{


namespace math
{


//--------------------------------------------------------------------------------------------------
// matrix4 members
//

template < typename T >
VV_FORCE_INLINE matrix< 4, 4, T >::matrix()
{
}

template < typename T >
VV_FORCE_INLINE matrix< 4, 4, T >::matrix
(
    vector< 4, T > const& c0,
    vector< 4, T > const& c1,
    vector< 4, T > const& c2,
    vector< 4, T > const& c3
)
    : col0(c0)
    , col1(c1)
    , col2(c2)
    , col3(c3)
{
}

template < typename T >
VV_FORCE_INLINE matrix< 4, 4, T >::matrix
(
    T const& m00, T const& m10, T const& m20, T const& m30,
    T const& m01, T const& m11, T const& m21, T const& m31,
    T const& m02, T const& m12, T const& m22, T const& m32,
    T const& m03, T const& m13, T const& m23, T const& m33
)
    : col0(m00, m10, m20, m30)
    , col1(m01, m11, m21, m31)
    , col2(m02, m12, m22, m32)
    , col3(m03, m13, m23, m33)
{
}

template < typename T >
VV_FORCE_INLINE matrix< 4, 4, T >::matrix(T const data[16])
    : col0(&data[ 0])
    , col1(&data[ 4])
    , col2(&data[ 8])
    , col3(&data[12])
{
}

template < typename T >
VV_FORCE_INLINE T* matrix< 4, 4, T >::data()
{
    return reinterpret_cast< T* >(this);
}

template < typename T >
VV_FORCE_INLINE T const* matrix< 4, 4, T >::data() const
{
    return reinterpret_cast< T const* >(this);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T >& matrix< 4, 4, T >::operator()(size_t col)
{
    return *(reinterpret_cast< vector< 4, T >* >(this) + col);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > const& matrix< 4, 4, T >::operator()(size_t col) const
{
    return *(reinterpret_cast< vector< 4, T > const* >(this) + col);
}

template < typename T >
VV_FORCE_INLINE T& matrix< 4, 4, T >::operator()(size_t row, size_t col)
{
    return (operator()(col))[row];
}

template < typename T >
VV_FORCE_INLINE T const& matrix< 4, 4, T >::operator()(size_t row, size_t col) const
{
    return (operator()(col))[row];
}
    

//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template < typename T >
VV_FORCE_INLINE matrix< 4, 4, T > operator*(matrix< 4, 4, T > const& a, matrix< 4, 4, T > const& b)
{

    return matrix< 4, 4, T >
    (
        a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0) + a(0, 2) * b(2, 0) + a(0, 3) * b(3, 0),
        a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0) + a(1, 2) * b(2, 0) + a(1, 3) * b(3, 0),
        a(2, 0) * b(0, 0) + a(2, 1) * b(1, 0) + a(2, 2) * b(2, 0) + a(2, 3) * b(3, 0),
        a(3, 0) * b(0, 0) + a(3, 1) * b(1, 0) + a(3, 2) * b(2, 0) + a(3, 3) * b(3, 0),
        a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1) + a(0, 2) * b(2, 1) + a(0, 3) * b(3, 1),
        a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1) + a(1, 2) * b(2, 1) + a(1, 3) * b(3, 1),
        a(2, 0) * b(0, 1) + a(2, 1) * b(1, 1) + a(2, 2) * b(2, 1) + a(2, 3) * b(3, 1),
        a(3, 0) * b(0, 1) + a(3, 1) * b(1, 1) + a(3, 2) * b(2, 1) + a(3, 3) * b(3, 1),
        a(0, 0) * b(0, 2) + a(0, 1) * b(1, 2) + a(0, 2) * b(2, 2) + a(0, 3) * b(3, 2),
        a(1, 0) * b(0, 2) + a(1, 1) * b(1, 2) + a(1, 2) * b(2, 2) + a(1, 3) * b(3, 2),
        a(2, 0) * b(0, 2) + a(2, 1) * b(1, 2) + a(2, 2) * b(2, 2) + a(2, 3) * b(3, 2),
        a(3, 0) * b(0, 2) + a(3, 1) * b(1, 2) + a(3, 2) * b(2, 2) + a(3, 3) * b(3, 2),
        a(0, 0) * b(0, 3) + a(0, 1) * b(1, 3) + a(0, 2) * b(2, 3) + a(0, 3) * b(3, 3),
        a(1, 0) * b(0, 3) + a(1, 1) * b(1, 3) + a(1, 2) * b(2, 3) + a(1, 3) * b(3, 3),
        a(2, 0) * b(0, 3) + a(2, 1) * b(1, 3) + a(2, 2) * b(2, 3) + a(2, 3) * b(3, 3),
        a(3, 0) * b(0, 3) + a(3, 1) * b(1, 3) + a(3, 2) * b(2, 3) + a(3, 3) * b(3, 3)
    );

}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator*(matrix< 4, 4, T > const& m, vector< 4, T > const& v)
{

    return vector< 4, T >
    (
        m(0, 0) * v.x + m(0, 1) * v.y + m(0, 2) * v.z + m(0, 3) * v.w,
        m(1, 0) * v.x + m(1, 1) * v.y + m(1, 2) * v.z + m(1, 3) * v.w,
        m(2, 0) * v.x + m(2, 1) * v.y + m(2, 2) * v.z + m(2, 3) * v.w,
        m(3, 0) * v.x + m(3, 1) * v.y + m(3, 2) * v.z + m(3, 3) * v.w
    );

}


} // math


} // virvo


