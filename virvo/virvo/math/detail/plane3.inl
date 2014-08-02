namespace virvo
{


//--------------------------------------------------------------------------------------------------
// plnae3 members
//

template < typename T >
inline hyper_plane< 3, T >::hyper_plane()
{
}

template < typename T >
inline hyper_plane< 3, T >::hyper_plane(vector< 3, T > const& n, T o)
    : normal(n)
    , offset(o)
{
}

template < typename T >
inline hyper_plane< 3, T >::hyper_plane(vector< 3, T > const& n, vector< 3, T > const& p)
    : normal(n)
    , offset(-dot(n, p))
{
}

} // virvo


