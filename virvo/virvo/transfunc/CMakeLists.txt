find_package(Boost COMPONENTS filesystem serialization system REQUIRED)

deskvox_use_package(Boost)
if(DESKVOX_USE_CUDA)
find_package(CUDA)
deskvox_use_package(CUDA)
endif()

set(VIRVO_TRANSFUNC_HEADERS
    ${VIRVO_SOURCE_DIR}/vvcolor.h
    ${VIRVO_SOURCE_DIR}/vvdebugmsg.h
    ${VIRVO_SOURCE_DIR}/vvtfwidget.h
    ${VIRVO_SOURCE_DIR}/vvtoolshed.h
    ${VIRVO_SOURCE_DIR}/vvtransfunc.h
)

set(VIRVO_TRANSFUNC_SOURCES
    ${VIRVO_SOURCE_DIR}/private/vvlog.cpp
    ${VIRVO_SOURCE_DIR}/vvcolor.cpp
    ${VIRVO_SOURCE_DIR}/vvdebugmsg.cpp
    ${VIRVO_SOURCE_DIR}/vvtfwidget.cpp
    ${VIRVO_SOURCE_DIR}/vvtoolshed.cpp
    ${VIRVO_SOURCE_DIR}/vvtransfunc.cpp
)

if(DESKVOX_USE_CUDA)
    set(VIRVO_TRANSFUNC_SOURCES ${VIRVO_TRANSFUNC_SOURCES}
        ${VIRVO_SOURCE_DIR}/vvcudatransfunc.cpp
    )
deskvox_cuda_compiles(VIRVO_TRANSFUNC_CUDA_SOURCES
    ${VIRVO_SOURCE_DIR}/vvcudatransfunc.cu
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
