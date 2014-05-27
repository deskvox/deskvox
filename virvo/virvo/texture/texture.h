#ifndef VV_TEXTURE_H
#define VV_TEXTURE_H


#include "detail/sampler3d.h"
#include "detail/texture3d.h"



namespace virvo
{


template < int bpc, typename VoxelT, typename FloatT >
VV_FORCE_INLINE FloatT tex3D(texture< VoxelT, NormalizedFloat, 3 > const& tex, math::base_vec3< FloatT > coord)
{

    math::base_vec3< FloatT >  size( tex.width(), tex.height(), tex.depth() );
    return detail::tex3D< bpc >( tex.data, coord, size, tex.get_filter_mode() );

}


} // virvo


#endif // VV_TEXTURE_H


