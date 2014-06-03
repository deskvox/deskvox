#ifndef VV_MATH_FORWARD_H
#define VV_MATH_FORWARD_H


#include <cstddef>


namespace virvo
{


namespace math
{

//--------------------------------------------------------------------------------------------------
// Declarations
//

template < size_t Dim, typename T >
class vector;

template < size_t N /* rows */, size_t M /* columns */, typename T >
class matrix;


//--------------------------------------------------------------------------------------------------
// Most common typedefs
//

typedef vector< 2, int >                vec2i;
typedef vector< 2, unsigned int >       vec2ui;
typedef vector< 2, float >              vec2f;
typedef vector< 2, double >             vec2d;


typedef vector< 3, int >                vec3i;
typedef vector< 3, unsigned int >       vec3ui;
typedef vector< 3, float >              vec3f;
typedef vector< 3, double >             vec3d;


typedef vector< 4, int >                vec4i;
typedef vector< 4, unsigned int >       vec4ui;
typedef vector< 4, float >              vec4f;
typedef vector< 4, double >             vec4d;


} // math


} // virvo


#endif // VV_MATH_FORWARD_H


