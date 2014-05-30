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
VV_FORCE_INLINE math::vector< 4, math::simd::float4 > tex1D(texture< VoxelT, NormalizedFloat, 1 > const& tex, math::simd::float4 coord)
{

    using math::simd::float4;

    // special case for AoS rgba colors
    typedef math::vector< 4, float4 > return_type;

    float4 size = tex.width();
    return detail::tex1D< return_type >( tex.data, coord, size, tex.get_filter_mode() );

}


//-------------------------------------------------------------------------------------------------
// tex3D - general case and specializations
//

template < typename VoxelT, typename FloatT >
VV_FORCE_INLINE VoxelT tex3D(texture< VoxelT, NormalizedFloat, 3 > const& tex, math::vector< 3, FloatT > coord)
{

    // general case: return type equals voxel type
    typedef VoxelT return_type;

    math::vector< 3, FloatT >  size( tex.width(), tex.height(), tex.depth() );
    return detail::tex3D< return_type >( tex.data, coord, size, tex.get_filter_mode() );

}


template < typename VoxelT >
VV_FORCE_INLINE math::simd::float4 tex3D(texture< VoxelT, NormalizedFloat, 3 > const& tex, math::vector< 3, math::simd::float4 > coord)
{

    // special case: lookup four voxels at once and return as 32-bit float vector
    typedef math::simd::float4 return_type;

    math::vector< 3, math::simd::float4 > size( tex.width(), tex.height(), tex.depth() );
    return detail::tex3D< return_type >( tex.data, coord, size, tex.get_filter_mode() );

}


} // virvo


#endif // VV_TEXTURE_H


