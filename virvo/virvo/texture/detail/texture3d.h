#ifndef VV_TEXTURE_TEXTURE3D_H
#define VV_TEXTURE_TEXTURE3D_H


#include "texture_common.h"

#include <stddef.h>


namespace virvo
{


template
<
    typename T,
    tex_read_mode ReadMode
>
class texture< T, ReadMode, 3 > : public texture_base< T, ReadMode >
{
public:

    typedef texture_base< T, ReadMode > base_type;
    typedef typename base_type::value_type value_type;


    texture() {}

    texture(size_t w, size_t h, size_t d)
        : width_(w)
        , height_(h)
        , depth_(d)
    {
    }


    value_type& operator()(size_t x, size_t y, size_t z)
    {
        return base_type::data[z * width_ * height_ + y * width_ + x];
    }

    value_type const& operator()(size_t x, size_t y, size_t z) const
    {
        return base_type::data[z * width_ * height_ + y * width_ + x];
    }


    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t depth() const { return depth_; }

private:

    size_t width_;
    size_t height_;
    size_t depth_;

};


} // virvo


#endif // VV_TEXTURE_TEXTURE3D_H


