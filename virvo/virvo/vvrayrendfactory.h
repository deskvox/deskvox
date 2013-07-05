#pragma once

#include "vvexport.h"

/*! \brief  factory function, can e. g. be called by dlsym() to create rayrend from within plugin
 */
extern "C" VVAPI vvRenderer* createRayRend(vvVolDesc* vd, vvRenderState const& rs);

