// Symbol.h


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
