if(DESKVOX_USE_CUDA)

find_package(CUDA)
if(NOT CUDA_FOUND)
    return()
endif()

deskvox_use_package(CUDA INCDIRS ${CUDA_TOOLKIT_INCLUDE})

#deskvox_link_libraries(virvo)
#deskvox_link_libraries(virvo_fileio)

set(RAYREND_HEADERS
    ../../vvraycaster.h
)

add_definitions(-Dvirvo_EXPORTS)
add_definitions(-DHAVE_CONFIG_H)
add_definitions(-DVV_ARCH_CUDA=1)

if(DESKVOX_USE_OLD_CUDA)
    set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
    cuda_add_library(rayrendcuda ../../vvraycaster.cu)

    set_target_properties(rayrendcuda PROPERTIES LINKER_LANGUAGE CXX)
else()
    add_library(rayrendcuda ../../vvraycaster.cu)
endif()

target_link_libraries(rayrendcuda PUBLIC visionaray::visionaray virvo virvo_fileio)

# MSVC + CUDA: link with legacy stdio library

if (MSVC_VERSION GREATER 1800 AND DESKVOX_USE_CUDA AND CUDA_FOUND)
    target_link_libraries(rayrendcuda legacy_stdio_definitions.lib)
endif()

endif(DESKVOX_USE_CUDA)
