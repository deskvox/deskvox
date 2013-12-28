#pragma once


#include "vvaabb.h"
#include "vvrect.h"
#include "vvvecmath.h"


namespace virvo
{


/*! project from object coordinates to window coordinates
 *
 * @param modelview   modelview matrix in OpenGL order (column-major)
 * @param projection  projection matrix in OpenGL order (column-major)
 */
void project(Vec3* win, Vec3 const& obj, Matrix const& modelview, Matrix const& projection, Viewport const& viewport);


/*! unproject from window coordinates to project coordinates
 *
 * @param modelview   modelview matrix in OpenGL order (column-major)
 * @param projection  projection matrix in OpenGL order (column-major)
 */
void unproject(Vec3* obj, Vec3 const& win, Matrix const& modelview, Matrix const& projection, Viewport const& viewport);


/*! calc bounding rect of box in screen space coordinates
 */
template < typename T >
virvo::Recti bounds(vvBaseAABB< T > const& aabb, Matrix const& modelview, Matrix const& projection, Viewport const& viewport);


} // virvo


#include "project.impl.h"


