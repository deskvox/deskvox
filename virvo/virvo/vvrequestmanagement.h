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

#ifndef _VV_REQUESTMANAGEMENT_H_
#define _VV_REQUESTMANAGEMENT_H_

#include <string>
#include <vector>

#include "vvinttypes.h"
#include "vvrenderer.h"
#include "vvtcpsocket.h"

class vvGpu
{
public:
  struct vvGpuInfo
  {
    int freeMem;
    int totalMem;
  };

  /**
    Get a list of known gpus available from for this process configured in a config file.
    @return vector-list of vvGpus
    */
  static std::vector<vvGpu*> list();
  /**
    Get the current gpu infos of a gpus from either list() or createGpu()
    @return vvGpuInfo with values up to date or -1 if not available
    */
  static vvGpuInfo getInfo(vvGpu *gpu);
  /**
    Create a gpu object from a configured string with notation like "key=value,key=value,..."
    @return corresponding vvGpu object or NULL on error
    */
  static vvGpu* createGpu(std::string& data);

private:
  vvGpu();
  vvGpu(const vvGpu& rhs);
  vvGpu& operator = (const vvGpu& src);

  class GpuData;

  GpuData *_data;
};

struct vvRequest
{
  vvRequest()
    : type(vvRenderer::TEXREND)
    , niceness(0)
    , sock(NULL)
  {
    nodes.push_back(1);
  }

  vvRenderer::RendererType type;  ///< requested rendering type
  int         niceness;           ///< niceness priority ranging from -20 to 20
  typedef int numgpus;
  std::vector<numgpus> nodes;     ///< requested amount of nodes with corresponding number of gpus

  vvTcpSocket *sock;              ///< socket to requesting client

  bool operator<(vvRequest other)
  {
    if(niceness != other.niceness)
    {
      return niceness < other.niceness;
    }
    else
    {
      // sort more note-requests first
      return nodes.size() > other.nodes.size();
    }
  }
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
