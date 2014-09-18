#ifndef VSNRAY_MATH_QUAT_H
#define VSNRAY_MATH_QUAT_H

#include <stddef.h>

namespace MATH_NAMESPACE
{

class quat
{
public:

    float w;
    float x;
    float y;
    float z;

    quat();
    quat(float w, float x, float y, float z);
    quat(float w, vec3 const& v);

    static quat identity();

};

} // visionaray

#include "detail/quat.inl"

#endif // VSNRAY_MATH_QUAT_H


