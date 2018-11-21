// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <virvo/vvclock.h>

template <typename Tex>
void KdTree::updateTransfunc(Tex transfunc)
{
  using namespace visionaray;

#ifdef BUILD_TIMING
  vvStopwatch sw; sw.start();
#endif
  psvt.build(transfunc);
#ifdef BUILD_TIMING
  std::cout << std::fixed << std::setprecision(3) << "svt update: " << sw.getTime() << " sec.\n";
#endif

#ifdef BUILD_TIMING
  sw.start();
#endif
  root.reset(new Node);
  root->bbox = psvt.boundary(aabbi(vec3i(0), vec3i(vox[0], vox[1], vox[2])));
  root->depth = 0;
  node_splitting(root);
#ifdef BUILD_TIMING
  std::cout << "splitting: " << sw.getTime() << " sec.\n";
#endif
}
