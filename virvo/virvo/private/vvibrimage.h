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


#ifndef VV_PRIVATE_IBR_IMAGE_H
#define VV_PRIVATE_IBR_IMAGE_H


#include "vvimage.h"
#include "vvrect.h" // virvo::Viewport
#include "vvvecmath.h"


class vvSocketIO;


namespace virvo
{


class IbrImage : public Image
{
  friend class ::vvSocketIO; // Serialize/Deserialize

  typedef Image BaseType;

public:
  // Construct an empty (invalid) IBR image
  VVAPI IbrImage();

  // Construct a new IBR image
  VVAPI IbrImage(int w, int h, int colorBits, int depthBits);

  // Destructor
  VVAPI virtual ~IbrImage();

  // Returns a pointer to the depth buffer
  unsigned char* depthData() { return &depth_[0]; }

  // Returns a pointer to the depth buffer
  unsigned char const* depthData() const { return &depth_[0]; }

  // Returns the size of the depth buffer
  size_t depthSize() const
  {
    // size mismatch? internal error
    assert(depth_.size() == static_cast<size_t>(width() * height() * (depthBits() / 8)));

    return depth_.size();
  }

  // Returns the number of bits per color value
  int colorBits() const { return 8 * pixlen(); }

  // Returns the number of bits per depth value
  int depthBits() const { return depthBits_; }

  // Returns the minimum depth value
  float& depthMin() { return depthMin_; }

  // Returns the minimum depth value
  float const& depthMin() const { return depthMin_; }

  // Returns the maximum depth value
  float& depthMax() { return depthMax_; }

  // Returns the maximum depth value
  float const& depthMax() const { return depthMax_; }

  // Returns the model-view matrix
  vvMatrix& viewMatrix() { return viewMatrix_; }

  // Returns the model-view matrix
  vvMatrix const& viewMatrix() const { return viewMatrix_; }

  // Returns the projection matrix
  vvMatrix& projMatrix() { return projMatrix_; }

  // Returns the projection matrix
  vvMatrix const& projMatrix() const { return projMatrix_; }

  // Returns the viewport
  virvo::Viewport& viewport() { return viewport_; }

  // Returns the viewport
  virvo::Viewport const& viewport() const { return viewport_; }

  // Compress the image
  VVAPI bool compress();

  // Decompress the image
  VVAPI bool decompress();

private:
  // The depth buffer
  std::vector<unsigned char> depth_;
  // Precision of the depth buffer
  int depthBits_;
  // Depth range
  float depthMin_;
  float depthMax_;
  // View matrix
  vvMatrix viewMatrix_;
  // Projection matrix
  vvMatrix projMatrix_;
  // The viewport
  virvo::Viewport viewport_;
};


} // namespace virvo


#endif // VV_PRIVATE_IBR_IMAGE_H
