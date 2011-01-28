// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
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

#ifndef VV_CUDA_H
#define VV_CUDA_H

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CUDA
#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime_api.h>

class vvCuda
{
    public:
    static bool checkError(bool *success, cudaError_t err, const char *msg = NULL, bool syncIfDebug = true);
    static bool init();
    static bool initGlInterop();
    private:
    bool s_useGlInterop();
};
#endif

#endif // VV_CUDA_H

//============================================================================
// End of File
//============================================================================

