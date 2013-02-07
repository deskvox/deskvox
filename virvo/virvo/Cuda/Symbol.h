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

#pragma once


#include <cuda_runtime_api.h>


namespace virvo
{
namespace cuda
{


    template<class T>
    class Symbol
    {
        T* symbolPtr;

    public:
        Symbol(T* symbolPtr) : symbolPtr(symbolPtr)
        {
        }

        // Returns a pointer to the symbol
        T* get() { return symbolPtr; }

        // Returns a pointer to the symbol
        const T* get() const { return symbolPtr; }

        // Copy data to the given symbol on the device.
        template<class U>
        bool update(const U* hostPtr, size_t count, size_t offset = 0) const
        {
            return cudaSuccess == cudaMemcpyToSymbol(Name(), hostPtr, count, offset);
        }

        // Find the address associated with a CUDA symbol
        void* address() const
        {
            void* devPtr;

            if (cudaSuccess == cudaGetSymbolAddress(&devPtr, Name()))
                return devPtr;

            return 0;
        }

        // Find the size of the object associated with a CUDA symbol.
        size_t size() const
        {
            size_t symbolSize = 0;

            if (cudaSuccess == cudaGetSymbolSize(&symbolSize, Name()))
                return symbolSize;

            return 0;
        }

    private:
        // CUDA 5 Release Notes:
        // The use of a character string to indicate a device symbol, which was
        // possible with certain API functions, is no longer supported. Instead,
        // the symbol should be used directly.
#if CUDART_VERSION < 5000
        const char* Name() const { return reinterpret_cast<const char*>(symbolPtr); }
#else
        T* Name() const { return symbolPtr; }
#endif
    };


} // namespace cuda
} // namespace virvo
