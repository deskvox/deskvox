#ifndef VV_TEXTURE_COMMON_H
#define VV_TEXTURE_COMMON_H


namespace virvo
{


enum tex_filter_mode
{
    Nearest = 0,
    Linear,
    BSpline,
    BSplineInterpol
};


template < typename VoxelT, int Dim >
class texture;

}


#endif


