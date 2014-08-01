#ifndef VV_MATH_FORWARD_H
#define VV_MATH_FORWARD_H


#include <cstddef>


namespace virvo
{


//--------------------------------------------------------------------------------------------------
// Declarations
//

template < int Dim >
class cartesian_axis;

template < size_t Dim, typename T >
class vector;

template < size_t N /* rows */, size_t M /* columns */, typename T >
class matrix;

template < typename T >
class base_aabb;

template
<
    template < typename > class L,
    typename T
>
class rectangle;

template< typename T >
class xywh_layout;


//--------------------------------------------------------------------------------------------------
// Most common typedefs
//

typedef vector< 2, int >                    vec2i;
typedef vector< 2, unsigned int >           vec2ui;
typedef vector< 2, float >                  vec2f;
typedef vector< 2, double >                 vec2d;
typedef vector< 2, float >                  vec2;


typedef vector< 3, int >                    vec3i;
typedef vector< 3, unsigned int >           vec3ui;
typedef vector< 3, float >                  vec3f;
typedef vector< 3, double >                 vec3d;
typedef vector< 3, float >                  vec3;


typedef vector< 4, int >                    vec4i;
typedef vector< 4, unsigned int >           vec4ui;
typedef vector< 4, float >                  vec4f;
typedef vector< 4, double >                 vec4d;
typedef vector< 4, float >                  vec4;


typedef matrix< 4, 4, float >               mat4f;
typedef matrix< 4, 4, double >              mat4d;
typedef matrix< 4, 4, float >               mat4;


typedef base_aabb< int >                    aabbi;
typedef base_aabb< float >                  aabbf;
typedef base_aabb< double >                 aabbd;
typedef base_aabb< float >                  aabb;


typedef rectangle< xywh_layout, int >       recti;
typedef rectangle< xywh_layout, float >     rectf;
typedef rectangle< xywh_layout, double >    rectd;
typedef rectangle< xywh_layout, float >     rect;


} // virvo


#endif // VV_MATH_FORWARD_H


