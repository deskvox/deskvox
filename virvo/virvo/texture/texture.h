#ifndef VV_TEXTURE_H
#define VV_TEXTURE_H


#include "detail/sampler1d.h"
#include "detail/sampler3d.h"
#include "detail/texture1d.h"
#include "detail/texture3d.h"



namespace virvo
{

//-------------------------------------------------------------------------------------------------
// tex1D - general case and specializations
//

template < typename VoxelT, typename FloatT >
VV_FORCE_INLINE VoxelT tex1D(texture< VoxelT, NormalizedFloat, 1 > const& tex, FloatT coord)
{

    // general case: return type equals voxel type
    typedef VoxelT return_type;

    FloatT size = tex.width();
    return detail::tex1D< return_type >( tex.data, coord, size, tex.get_filter_mode() );

}


template < typename VoxelT >
VV_FORCE_INLINE math::base_vec4< math::sse_vec > tex1D(texture< VoxelT, NormalizedFloat, 1 > const& tex, math::sse_vec coord)
{

    // special case for AoS rgba colors
    typedef math::base_vec4< math::sse_vec > return_type;

    math::sse_vec size = tex.width();
    return detail::tex1D< return_type >( tex.data, coord, size, tex.get_filter_mode() );

}


//-------------------------------------------------------------------------------------------------
// tex3D - general case and specializations
//

template < typename VoxelT, typename FloatT >
VV_FORCE_INLINE VoxelT tex3D(texture< VoxelT, NormalizedFloat, 3 > const& tex, math::base_vec3< FloatT > coord)
{

    // general case: return type equals voxel type
    typedef VoxelT return_type;

    math::base_vec3< FloatT >  size( tex.width(), tex.height(), tex.depth() );
    return detail::tex3D< return_type >( tex.data, coord, size, tex.get_filter_mode() );

}


template < typename VoxelT >
VV_FORCE_INLINE math::sse_vec tex3D(texture< VoxelT, NormalizedFloat, 3 > const& tex, math::base_vec3< math::sse_vec > coord)
{

    // special case: lookup four voxels at once and return as 32-bit float vector
    typedef math::sse_vec return_type;

    math::base_vec3< math::sse_vec > size( tex.width(), tex.height(), tex.depth() );
    return detail::tex3D< return_type >( tex.data, coord, size, tex.get_filter_mode() );

}


} // virvo


#endif // VV_TEXTURE_H


