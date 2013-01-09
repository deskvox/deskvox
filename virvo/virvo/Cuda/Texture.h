// Texture.h


#pragma once


#include "Array.h"

#include <assert.h>


namespace virvo
{
namespace cuda
{


    class Texture
    {
        textureReference* texPtr;

    public:
        Texture(textureReference* texPtr) : texPtr(texPtr)
        {
        }

        void setNormalized(int normalized) const
        {
            texPtr->normalized = normalized;
        }

        void setFilterMode(cudaTextureFilterMode mode) const
        {
            texPtr->filterMode = mode;
        }

        void setAddressMode(size_t index, cudaTextureAddressMode mode) const
        {
            assert( index < 3 ); // index out of range
            texPtr->addressMode[index] = mode;
        }

        void setAddressMode(cudaTextureAddressMode mode) const
        {
            setAddressMode(0, mode);
            setAddressMode(1, mode);
            setAddressMode(2, mode);
        }

        void setChannelFormatDesc(const cudaChannelFormatDesc& desc) const
        {
            texPtr->channelDesc = desc;
        }

        void setSRGB(int sRGB) const
        {
            texPtr->sRGB = sRGB;
        }

        // Returns the current channel descriptor
        const cudaChannelFormatDesc& desc() const
        {
            return texPtr->channelDesc;
        }

        // Binds a memory area to a texture
        template<class T>
        bool bind(size_t* offset, const T* devPtr, const cudaChannelFormatDesc& desc, size_t size = size_t(-1)) const
        {
            return cudaSuccess == cudaBindTexture(offset, texPtr, devPtr, &desc, size);
        }

        // Binds a memory area to a texture
        // The channel descriptor is inherited from the texture reference
        template<class T>
        bool bind(size_t* offset, const T* devPtr, size_t size = size_t(-1)) const
        {
            return bind(offset, devPtr, texPtr->channelDesc, size);
        }

        // Binds a 2D memory area to a texture
        template<class T>
        bool bind(size_t* offset, const T* devPtr, const cudaChannelFormatDesc& desc, size_t width, size_t height, size_t pitch) const
        {
            return cudaSuccess == cudaBindTexture2D(offset, texPtr, devPtr, &desc, width, height, pitch);
        }

        // Binds a 2D memory area to a texture
        // The channel descriptor is inherited from the texture reference
        template<class T>
        bool bind(size_t* offset, const T* devPtr, size_t width, size_t height, size_t pitch) const
        {
            return bind(offset, devPtr, texPtr->channelDesc, width, height, pitch);
        }

        // Binds an array to a texture
        bool bind(cudaArray_const_t arr, const cudaChannelFormatDesc& desc) const
        {
            return cudaSuccess == cudaBindTextureToArray(texPtr, arr, &desc);
        }

        // Binds an array to a texture
        // The channel descriptor is inherited from the texture reference
        bool bind(cudaArray_const_t arr) const
        {
            return bind(arr, texPtr->channelDesc);
        }

        // Binds an array to a texture
        bool bind(const Array& arr, const cudaChannelFormatDesc& desc) const
        {
            return bind(arr.get(), desc);
        }

        // Binds an array to a texture
        // The channel descriptor is inherited from the texture reference
        bool bind(const Array& arr) const
        {
            return bind(arr.get());
        }

        void unbind() const
        {
            cudaUnbindTexture(texPtr);
        }
    };


} // namespace cuda
} // namespace virvo
