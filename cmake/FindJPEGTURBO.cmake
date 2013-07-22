include(FindPackageHandleStandardArgs)

set(hints
  $ENV{EXTERNLIBS}/libjpeg-turbo
  $ENV{LIB_BASE_PATH}/libjpeg-turbo
)

set(paths
  /usr
  /usr/local
)

find_path(JPEGTURBO_INCLUDE_DIR
  NAMES
    jconfig.h jerror.h jmorecfg.h jpeglib.h # turbojpeg.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
)

find_library(JPEGTURBO_LIBRARY
  NAMES
    jpeg
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

find_library(JPEGTURBO_LIBRARY_DEBUG
  NAMES
    jpegd
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

if(NOT JPEGTURBO_LIBRARY_DEBUG)
  find_library(JPEGTURBO_LIBRARY_DEBUG
    NAMES
      jpeg
    HINTS
      ${hints}
    PATHS
      ${paths}
    PATH_SUFFIXES
      lib64/debug
      lib/debug
  )
endif()

if(JPEGTURBO_LIBRARY_DEBUG)
  set(JPEGTURBO_LIBRARIES optimized ${JPEGTURBO_LIBRARY} debug ${JPEGTURBO_LIBRARY_DEBUG})
else()
  set(JPEGTURBO_LIBRARIES ${JPEGTURBO_LIBRARY})
endif()

find_package_handle_standard_args(JPEGTURBO DEFAULT_MSG
  JPEGTURBO_INCLUDE_DIR
  JPEGTURBO_LIBRARY
)
