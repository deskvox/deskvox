find_package(Boost COMPONENTS filesystem serialization system REQUIRED)
find_package(GDCM)
find_package(Nifti)
find_package(Teem)

if (GDCM_FOUND)
    include(${GDCM_USE_FILE})
    add_definitions(-DHAVE_GDCM)
endif()

deskvox_use_package(Boost)
deskvox_use_package(GDCM)
deskvox_use_package(Nifti)
deskvox_use_package(Teem)

set(VIRVO_FILEIO_SOURCES
    ${VIRVO_SOURCE_DIR}/fileio/nifti.cpp
    ${VIRVO_SOURCE_DIR}/fileio/nrrd.cpp
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
)

deskvox_add_library(virvo_fileio
    ${VIRVO_FILEIO_SOURCES}
)
