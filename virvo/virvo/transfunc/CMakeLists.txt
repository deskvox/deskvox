find_package(Boost COMPONENTS filesystem serialization system REQUIRED)

deskvox_use_package(Boost)
if(DESKVOX_USE_CUDA)
find_package(CUDA)
deskvox_use_package(CUDA)
endif()

set(VIRVO_TRANSFUNC_HEADERS
    ../vvcolor.h
    ../vvdebugmsg.h
    ../vvtfwidget.h
    ../vvtoolshed.h
    ../vvtransfunc.h
)

set(VIRVO_TRANSFUNC_SOURCES
    ../private/vvlog.cpp
    ../vvcolor.cpp
    ../vvdebugmsg.cpp
    ../vvtfwidget.cpp
    ../vvtoolshed.cpp
    ../vvtransfunc.cpp
)

if(DESKVOX_USE_CUDA)
    set(VIRVO_TRANSFUNC_SOURCES ${VIRVO_TRANSFUNC_SOURCES}
        ../vvcudatransfunc.cpp
    )
deskvox_cuda_compiles(VIRVO_TRANSFUNC_CUDA_SOURCES
    ../vvcudatransfunc.cu
)
endif()

set(VIRVO_TRANSFUNC_DEFINITIONS
    HAVE_CONFIG_H
)

deskvox_add_library(virvo_transfunc
    ${VIRVO_TRANSFUNC_HEADERS}
    ${VIRVO_TRANSFUNC_SOURCES}
    ${VIRVO_TRANSFUNC_CUDA_SOURCES}
)

set_target_properties(virvo_transfunc
    PROPERTIES
    COMPILE_DEFINITIONS ${VIRVO_TRANSFUNC_DEFINITIONS}
)
