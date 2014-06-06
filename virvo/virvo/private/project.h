#pragma once


#include "math/math.h"

#include "vvaabb.h"


namespace virvo
{


/*! project from object coordinates to window coordinates
 *
 * @param modelview   modelview matrix in OpenGL order (column-major)
 * @param projection  projection matrix in OpenGL order (column-major)
 */
template < typename T >
void project(math::vector< 3, T >* win, math::vector< 3, T > const& obj, math::matrix< 4, 4, T > const& modelview,
    math::matrix< 4, 4, T > const& projection, math::recti const& viewport);


/*! unproject from window coordinates to project coordinates
 *
 * @param modelview   modelview matrix in OpenGL order (column-major)
 * @param projection  projection matrix in OpenGL order (column-major)
 */
template < typename T >
void unproject(math::vector< 3, T >* obj, math::vector< 3, T > const& win, math::matrix< 4, 4, T > const& modelview,
    math::matrix< 4, 4, T > const& projection, math::recti const& viewport);


/*! calc bounding rect of box in screen space coordinates
 */
template < typename T >
math::recti bounds(vvBaseAABB< T > const& aabb, math::matrix< 4, 4, T > const& modelview,
    math::matrix< 4, 4, T > const& projection, math::recti const& viewport);


} // virvo


#include "project.impl.h"


