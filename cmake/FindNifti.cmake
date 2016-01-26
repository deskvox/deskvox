include(FindPackageHandleStandardArgs)

set(hints
  $ENV{LIB_BASE_PATH}/nifti
)

set(paths
  /usr
  /usr/local
)

find_path(Nifti_INCLUDE_DIR
  NAMES
    nifti1.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
    include/nifti
)

find_library(Nifti_LIBRARY
  NAMES
    niftiio
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

find_library(Nifti_LIBRARY_DEBUG
  NAMES
    niftioiod
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

if(Nifti_LIBRARY_DEBUG)
  set(Nifti_LIBRARIES optimized ${Nifti_LIBRARY} debug ${Nifti_LIBRARY_DEBUG})
else()
  set(Nifti_LIBRARIES ${Nifti_LIBRARY})
endif()

find_package_handle_standard_args(Nifti
  DEFAULT_MSG
  Nifti_INCLUDE_DIR
  Nifti_LIBRARY
)
