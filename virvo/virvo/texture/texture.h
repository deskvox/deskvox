#ifndef VV_TEXTURE_H
#define VV_TEXTURE_H


#include "detail/sampler3d.h"
#include "detail/texture3d.h"



namespace virvo
{


template < int bpc, typename Int3T, typename Float3T, typename VoxelT >
VV_FORCE_INLINE typename Float3T::value_type tex3D(texture< VoxelT, 3 > const& tex, Float3T coord)
{

    Int3T size( tex.width(), tex.height(), tex.depth() );
    return detail::tex3D< bpc >( tex.data, coord, size, tex.get_filter_mode() );

}


} // virvo


#endif // VV_TEXTURE_H


