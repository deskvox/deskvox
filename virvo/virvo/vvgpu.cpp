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

#include "vvdebugmsg.h"
#include "vvgpu.h"
#include "vvrendercontext.h"
#include "vvtoolshed.h"

#include <algorithm>
#include <GL/glew.h>
#include <sstream>
#include <istream>
#include <fstream>
#include <string>

#define GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX 0x9048
#define GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX 0x9049

class vvGpu::GpuData
{
public:
  GpuData()
  {
    glName = "";
    Xdsp   = "";
    cuda   = false;
    openGL = false;
    cudaDevice = -1;
    wSystem = vvRenderContext::VV_NONE;
  }
  std::string glName;
  std::string Xdsp;
  bool        cuda;
  bool        openGL;
  int         cudaDevice;
  vvRenderContext::WindowingSystem wSystem;
};

std::vector<vvGpu*> vvGpu::list()
{
  std::vector<vvGpu*> gpus;

  std::ifstream fin("/raid/home/sdelisav/deskvox/qtcreator-build/virvo/tools/vserver/vserver.config");

  if(!fin.is_open())
  {
    vvDebugMsg::msg(2, "vvGpu::list() could not open config file");
  }

  uint lineNum = 0;
  std::string line;
  while(fin.good())
  {
    lineNum++;
    std::getline(fin, line);

    std::vector<std::string> subStrs = vvToolshed::split(line, "=");
    if(subStrs.size() < 2)
    {
      vvDebugMsg::msg(2, "vvGpu::list() nothing to parse in config file line ", (int)lineNum);
    }
    else
    {
      if(vvToolshed::strCompare("gpu", subStrs[0].c_str()) == 0)
      {
        line.erase(0,line.find_first_of("=",0)+1);
        gpus.push_back(vvGpu::createGpu(line));
      }
      else if(vvToolshed::strCompare("node", subStrs[0].c_str()) == 0)
      {
        // NODE bla bla
      }
    }
  }

  // TODO: add all found gpus here

  return gpus;
}

vvGpu::vvGpuInfo vvGpu::getInfo(vvGpu *gpu)
{
  vvGpuInfo inf = { -1, -1 };

  if(gpu->_data->openGL)
  {
    vvContextOptions co;
    co.displayName = gpu->_data->Xdsp;
    co.doubleBuffering = false;
    co.height = 1;
    co.width = 1;
    co.type = vvContextOptions::VV_WINDOW;

    vvRenderContext context = vvRenderContext(co);
    context.makeCurrent();

    glGetIntegerv(GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX,
                  &(inf.totalMem));

    glGetIntegerv(GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX,
                  &(inf.freeMem));
  }
  else if(gpu->_data->cuda)
  {
    // TODO: Implement cuda-case here!
  }

  return inf;
}

vvGpu* vvGpu::createGpu(std::string& data)
{
  vvDebugMsg::msg(3, "vvGpu::createGpu() Enter");

  std::vector<std::string> attributes = vvToolshed::split(data, ",");

  if(attributes.size() > 0)
  {
    vvGpu* gpu = new vvGpu();

    for(std::vector<std::string>::iterator attrib = attributes.begin(); attrib != attributes.end(); attrib++)
    {
      std::vector<std::string> nameValue = vvToolshed::split(*attrib, "=");

      if(nameValue.size() < 2)
      {
        vvDebugMsg::msg(0, "vvGpu::parseGpuData() parse error: attribute value missing");
        continue;
      }

      const char* attribNames[] =
      {
        "name",           // 1
        "xdsp",           // 2
        "cuda",           // 3
        "opengl",         // 4
        "windowingsystem" // 5
      };

      uint attrName = std::find(attribNames, attribNames+5, (nameValue[0])) - attribNames;
      attrName = (attrName < 12) ? (attrName + 1) : 0;

      switch(attrName)
      {
      case 1:
        gpu->_data->glName = nameValue[1];
        break;
      case 2:
        gpu->_data->Xdsp = nameValue[1];
        break;
      case 3:
        gpu->_data->cuda = vvToolshed::strCompare(nameValue[1].c_str(), "true") == 0 ? true : false;
        break;
      case 4:
        gpu->_data->openGL = vvToolshed::strCompare(nameValue[1].c_str(), "true") == 0 ? true : false;
        break;
      case 5:
        if(vvToolshed::strCompare(nameValue[1].c_str(), "X11"))
        {
          gpu->_data->wSystem = vvRenderContext::VV_X11;
        }
        else if(vvToolshed::strCompare(nameValue[1].c_str(), "WGL"))
        {
          gpu->_data->wSystem = vvRenderContext::VV_WGL;
        }
        else if(vvToolshed::strCompare(nameValue[1].c_str(), "COCOA"))
        {
          gpu->_data->wSystem = vvRenderContext::VV_COCOA;
        }
        else
        {
          vvDebugMsg::msg(2, "vvGpu::parseGpuData() parse error: unknown windowingsystem type");
          gpu->_data->wSystem = vvRenderContext::VV_NONE;
        }
        break;
      default:
        vvDebugMsg::msg(2, "vvGpu::createGpu() parse error: unknown attribute");
        delete gpu;
        return NULL;
      }
    }
    return gpu;
  }
  else
  {
    return NULL;
  }
}

vvGpu::vvGpu()
{
  _data = new GpuData;
}

vvGpu& vvGpu::operator = (const vvGpu& src)
{
  (void)src;
  return *this;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
