#pragma once


#include "math/math.h"

#include "vvaabb.h"
#include "vvrect.h"


namespace virvo
{


/*! project from object coordinates to window coordinates
 *
 * @param modelview   modelview matrix in OpenGL order (column-major)
 * @param projection  projection matrix in OpenGL order (column-major)
 */
void project(math::vec3f* win, math::vec3f const& obj, Matrix const& modelview, Matrix const& projection, Viewport const& viewport);


/*! unproject from window coordinates to project coordinates
 *
 * @param modelview   modelview matrix in OpenGL order (column-major)
 * @param projection  projection matrix in OpenGL order (column-major)
 */
void unproject(math::vec3f* obj, math::vec3f const& win, Matrix const& modelview, Matrix const& projection, Viewport const& viewport);


/*! calc bounding rect of box in screen space coordinates
 */
template < typename T >
virvo::Recti bounds(vvBaseAABB< T > const& aabb, Matrix const& modelview, Matrix const& projection, Viewport const& viewport);


} // virvo


#include "project.impl.h"


