#ifndef VV_TEXTURE_COMMON_H
#define VV_TEXTURE_COMMON_H


#include <vector>


#define VV_COMMON_TEX_IFACE                                                                 \
    enum tex_filter_mode                                                                    \
    {                                                                                       \
        Nearest = 0,                                                                        \
        Linear,                                                                             \
        Cubic,                                                                              \
        CubicInterpol                                                                       \
    };                                                                                      \
    typedef VoxelT value_type;                                                              \
    typedef std::vector< value_type > vec_type;                                             \
    typedef typename vec_type::iterator iterator;                                           \
    typedef typename vec_type::const_iterator const_iterator;                               \
    typename vec_type::iterator begin()                         { return data.begin(); }    \
    typename vec_type::const_iterator begin() const             { return data.begin(); }    \
    typename vec_type::iterator end()                           { return data.end(); }      \
    typename vec_type::const_iterator end() const               { return data.end(); }      \
    void clear()                                                { data.clear(); }           \
    value_type& operator[](size_t i)                            { return data[i]; }         \
    value_type const& operator[](size_t i) const                { return data[i]; }         \
    vec_type data;                                                                          \



namespace virvo
{

template < typename VoxelT, int Dim >
class texture;

}


#endif


