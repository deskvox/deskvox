find_package(Boost COMPONENTS serialization REQUIRED)
find_package(Nifti)
find_package(Teem)

if(DESKVOX_USE_GDCM)
    if (GDCM_FOUND)
        include(${GDCM_USE_FILE})
    endif()
endif()

deskvox_use_package(Boost)
if(DESKVOX_USE_CUDA)
    if(DESKVOX_CUDA_CMAKE)
        deskvox_use_package(CUDA INCDIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    else()
        deskvox_use_package(CUDA INCDIRS ${CUDA_TOOLKIT_INCLUDE})
    endif()
endif()
if(DESKVOX_USE_GDCM)
deskvox_use_package(GDCM)
endif()
deskvox_use_package(Nifti)
deskvox_use_package(Teem)
deskvox_use_package(cfitsio)

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

    gdcm.h
    feature.h
    nifti.h
    nrrd.h
    fits.h
)

set(VIRVO_FILEIO_SOURCES
    ${VIRVO_SOURCE_DIR}/private/vvlog.cpp
    ${VIRVO_SOURCE_DIR}/vvclock.cpp
    ${VIRVO_SOURCE_DIR}/vvdicom.cpp
    ${VIRVO_SOURCE_DIR}/vvfileio.cpp
    ${VIRVO_SOURCE_DIR}/vvtokenizer.cpp
    ${VIRVO_SOURCE_DIR}/vvvoldesc.cpp

    gdcm.cpp
    feature.cpp
    nifti.cpp
    nrrd.cpp
    fits.cpp
)

if(DESKVOX_USE_CUDA)
  set(VIRVO_FILEIO_SOURCES ${VIRVO_FILEIO_SOURCES}
    ${VIRVO_SOURCE_DIR}/vvcudatransfunc.cpp
    )
  deskvox_cuda_compiles(VIRVO_FILEIO_CUDA_SOURCES
    ${VIRVO_SOURCE_DIR}/vvcudatransfunc.cu
    )
endif()

set(VIRVO_FILEIO_DEFINITIONS
    HAVE_CONFIG_H
)
if(DESKVOX_USE_GDCM AND GDCM_FOUND)
   deskvox_link_libraries(gdcmDSED gdcmMSFF gdcmDICT gdcmCommon)
endif()

deskvox_link_libraries(virvo_transfunc)

message("file: sources ${VIRVO_FILEIO_SOURCES}")
deskvox_add_library(virvo_fileio
    ${VIRVO_FILEIO_HEADERS}
    ${VIRVO_FILEIO_SOURCES}
    ${VIRVO_FILEIO_CUDA_SOURCES}
)

set_target_properties(virvo_fileio
    PROPERTIES
        COMPILE_DEFINITIONS ${VIRVO_FILEIO_DEFINITIONS}
)
