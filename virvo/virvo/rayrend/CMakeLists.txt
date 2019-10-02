include(CheckCXXSourceCompiles)

find_package(Boost)
find_package(GLEW )
find_package(Visionaray)

if(NOT VISIONARAY_FOUND)
    message(STATUS "Visionaray not found, not building ray casting plugins")
    return()
endif()

deskvox_use_package(Boost)
deskvox_use_package(GLEW)
deskvox_use_package(VISIONARAY)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=0")
endif()

add_subdirectory(avx)
add_subdirectory(avx2)
add_subdirectory(cuda)
add_subdirectory(fpu)
add_subdirectory(sse2)
add_subdirectory(sse4_1)
