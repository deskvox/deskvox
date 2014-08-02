namespace virvo
{


template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator+=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u + v;
    return u;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator+=(vector< Dim, T >& v, T s)
{
    v = v + s;
    return v;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator-=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u - v;
    return u;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator-=(vector< Dim, T >& v, T s)
{
    v = v - s;
    return v;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator*=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u * v;
    return u;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator*=(vector< Dim, T >& v, T s)
{
    v = v * s;
    return v;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator/=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u / v;
    return u;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator/=(vector< Dim, T >& v, T s)
{
    v = v / s;
    return v;
}


} // virvo


