find_package(Boost COMPONENTS filesystem serialization system REQUIRED)
find_package(CUDA)
find_package(GDCM)
find_package(Nifti)
find_package(Teem)

if (GDCM_FOUND)
    include(${GDCM_USE_FILE})
    add_definitions(-DHAVE_GDCM)
endif()

deskvox_use_package(Boost)
if(DESKVOX_USE_CUDA)
deskvox_use_package(CUDA)
endif()
deskvox_use_package(GDCM)
deskvox_use_package(Nifti)
deskvox_use_package(Teem)

set(VIRVO_FILEIO_HEADERS
    ${VIRVO_SOURCE_DIR}/private/vvlog.h
    ${VIRVO_SOURCE_DIR}/vvclock.h
    ${VIRVO_SOURCE_DIR}/vvcolor.h
    ${VIRVO_SOURCE_DIR}/vvdebugmsg.h
    ${VIRVO_SOURCE_DIR}/vvdicom.h
    ${VIRVO_SOURCE_DIR}/vvfileio.h
    ${VIRVO_SOURCE_DIR}/vvtokenizer.h
    ${VIRVO_SOURCE_DIR}/vvtfwidget.h
    ${VIRVO_SOURCE_DIR}/vvtoolshed.h
    ${VIRVO_SOURCE_DIR}/vvtransfunc.h
    ${VIRVO_SOURCE_DIR}/vvvoldesc.h

    feature.h
    nifti.h
    nrrd.h
)


set(VIRVO_FILEIO_SOURCES
    ${VIRVO_SOURCE_DIR}/private/vvlog.cpp
    ${VIRVO_SOURCE_DIR}/vvclock.cpp
    ${VIRVO_SOURCE_DIR}/vvcolor.cpp
    ${VIRVO_SOURCE_DIR}/vvdebugmsg.cpp
    ${VIRVO_SOURCE_DIR}/vvdicom.cpp
    ${VIRVO_SOURCE_DIR}/vvfileio.cpp
    ${VIRVO_SOURCE_DIR}/vvtokenizer.cpp
    ${VIRVO_SOURCE_DIR}/vvtfwidget.cpp
    ${VIRVO_SOURCE_DIR}/vvtoolshed.cpp
    ${VIRVO_SOURCE_DIR}/vvtransfunc.cpp
    ${VIRVO_SOURCE_DIR}/vvvoldesc.cpp

    feature.cpp
    nifti.cpp
    nrrd.cpp
)

if(DESKVOX_USE_CUDA)
deskvox_cuda_compiles(VIRVO_FILEIO_CUDA_SOURCES
    ${VIRVO_SOURCE_DIR}/vvcudatransfunc.cu
)
endif()

if(GDCM_FOUND)
   deskvox_link_libraries(gdcmDSED gdcmMSFF gdcmDICT gdcmCommon gdcmuuid)
endif()

deskvox_add_library(virvo_fileio
    ${VIRVO_FILEIO_HEADERS}
    ${VIRVO_FILEIO_SOURCES}
    ${VIRVO_FILEIO_CUDA_SOURCES}
)
