#ifndef VV_TEXTURE_COMMON_H
#define VV_TEXTURE_COMMON_H


namespace virvo
{


enum tex_address_mode
{
    Wrap = 0,
    Mirror,
    Clamp,
    Border
};


enum tex_filter_mode
{
    Nearest = 0,
    Linear,
    BSpline,
    BSplineInterpol,
    CardinalSpline
};

enum tex_read_mode
{
    ElementType,
    NormalizedFloat
};


template
<
    typename VoxelT,
    tex_read_mode ReadMode
>
class texture_base
{
public:

    typedef VoxelT value_type;


    value_type* data;

    texture_base()
        : address_mode_(Wrap)
        , filter_mode_(Nearest)
    {
    }

    void set_address_mode(tex_address_mode mode) { address_mode_ = mode; }
    void set_filter_mode(tex_filter_mode mode) { filter_mode_ = mode; }

    tex_address_mode get_address_mode() const { return address_mode_; }
    tex_filter_mode get_filter_mode() const { return filter_mode_; }

protected:

    tex_address_mode address_mode_;
    tex_filter_mode filter_mode_;

};


template
<
    typename VoxelT,
    tex_read_mode ReadMode,
    int Dim
>
class texture;

}


#endif


