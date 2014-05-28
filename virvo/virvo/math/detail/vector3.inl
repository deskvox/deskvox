namespace virvo
{


namespace math
{


//--------------------------------------------------------------------------------------------------
// vector3 members
//

template < typename T >
VV_FORCE_INLINE vector< 3, T >::vector()
{
}

template < typename T >
VV_FORCE_INLINE vector< 3, T >::vector(T x, T y, T z)
    : x(x)
    , y(y)
    , z(z)
{
}

template < typename T >
VV_FORCE_INLINE vector< 3, T >::vector(T s)
    : x(s)
    , y(s)
    , z(s)
{
}

template < typename T >
template < typename U >
VV_FORCE_INLINE vector< 3, T >::vector(vector< 3, U > const& rhs)
    : x(rhs.x)
    , y(rhs.y)
    , z(rhs.z)
{
}

template < typename T >
template < typename U >
VV_FORCE_INLINE vector< 3, T >& vector< 3, T >::operator=(vector< 3, U > const& rhs)
{

    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    return *this;

}

template < typename T >
VV_FORCE_INLINE T* vector< 3, T >::data()
{
    return reinterpret_cast< T* >(this);
}

template < typename T >
VV_FORCE_INLINE T const* vector< 3, T >::data() const
{
    return reinterpret_cast< T const* >(this);
}

template < typename T >
VV_FORCE_INLINE T& vector< 3, T >::operator[](size_t i)
{
    return data()[i];
}

template < typename T >
VV_FORCE_INLINE T const& vector< 3, T >::operator[](size_t i) const
{
    return data()[i];
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator-(vector< 3, T > const& v)
{
    return vector< 3, T >(-v.x, -v.y, -v.z);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator+(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return vector< 3, T >(u.x + v.x, u.y + v.y, u.z + v.z);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator-(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return vector< 3, T >(u.x - v.x, u.y - v.y, u.z - v.z);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator*(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return vector< 3, T >(u.x * v.x, u.y * v.y, u.z * v.z);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator/(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return vector< 3, T >(u.x / v.x, u.y / v.y, u.z / v.z);
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template < typename T >
VV_FORCE_INLINE bool operator==(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return u.x == v.x && u.y == v.y && u.z == v.z;
}

template < typename T >
VV_FORCE_INLINE bool operator<(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return u.x < v.x || ( (u.x == v.x && u.y < v.y) || (u.y == v.y && u.z < v.z) );
}

template < typename T >
VV_FORCE_INLINE bool operator!=(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return !(u == v);
}

template < typename T >
VV_FORCE_INLINE bool operator<=(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return !(v < u);
}

template < typename T >
VV_FORCE_INLINE bool operator>(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return v < u;
}

template < typename T >
VV_FORCE_INLINE bool operator>=(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return !(u < v);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator+(vector< 3, T > const& v, T const& s)
{
    return vector< 3, T >(v.x + s, v.y + s, v.z + s);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator-(vector< 3, T > const& v, T const& s)
{
    return vector< 3, T >(v.x - s, v.y - s, v.z - s);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator*(vector< 3, T > const& v, T const& s)
{
    return vector< 3, T >(v.x * s, v.y * s, v.z * s);
}

template <typename T>
VV_FORCE_INLINE vector< 3, T > operator/(vector< 3, T > const& v, T const& s)
{
    return vector< 3, T >(v.x / s, v.y / s, v.z / s);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator+(T const& s, vector< 3, T > const& v)
{
    return vector< 3, T >(s + v.x, s + v.y, s + v.z);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator-(T const& s, vector< 3, T > const& v)
{
    return vector< 3, T >(s - v.x, s - v.y, s - v.z);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator*(T const& s, vector< 3, T > const& v)
{
    return vector< 3, T >(s * v.x, s * v.y, s * v.z);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > operator/(T const& s, vector< 3, T > const& v)
{
    return vector< 3, T >(s / v.x, s / v.y, s / v.z);
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template < typename T >
VV_FORCE_INLINE vector< 3, T > cross(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return vector< 3, T >
    (
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

template < typename T >
VV_FORCE_INLINE T dot(vector< 3, T > const& u, vector< 3, T > const& v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > norm(vector< 3, T >  const& v)
{
    return sqrt( dot(v, v) );
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > norm2(vector< 3, T > const& v)
{
    return dot(v, v);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > length(vector< 3, T > const& v)
{
    return norm(v);
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > normalize(vector< 3, T > const& v)
{
    return v * rsqrt( dot(v, v) );
}


} // math


} // virvo


