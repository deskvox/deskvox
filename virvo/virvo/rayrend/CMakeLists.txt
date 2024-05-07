include(CheckCXXSourceCompiles)

if(NOT DESKVOX_USE_VISIONARAY)
    message(STATUS "Visionaray disabled, not building ray casting plugins")
    return()
endif()

find_package(visionaray)

if(NOT TARGET visionaray::visionaray)
    message(STATUS "Visionaray not found, not building ray casting plugins")
    return()
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=0")
endif()

## add_subdirectory(avx)
## add_subdirectory(avx2)
add_subdirectory(fpu)
## add_subdirectory(sse2)
## add_subdirectory(sse4_1)
add_subdirectory(cuda)
