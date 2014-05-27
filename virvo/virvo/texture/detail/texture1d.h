#ifndef VV_TEXTURE_TEXTURE1D_H
#define VV_TEXTURE_TEXTURE1D_H


#include "texture_common.h"

#include <stddef.h>


namespace virvo
{


template
<
    typename T,
    tex_read_mode ReadMode
>
class texture< T, ReadMode, 1 > : public texture_base< T, ReadMode >
{
public:

    typedef texture_base< T, ReadMode > base_type;
    typedef typename base_type::value_type value_type;


    texture() {}

    texture(size_t w)
        : width_(w)
    {
    }


    value_type& operator()(size_t x)
    {
        return base_type::data[x];
    }

    value_type const& operator()(size_t x) const
    {
        return base_type::data[x];
    }


    size_t width() const { return width_; }

private:

    size_t width_;

};


} // virvo


#endif // VV_TEXTURE_TEXTURE1D_H


