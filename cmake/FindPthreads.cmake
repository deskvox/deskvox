include(FindPackageHandleStandardArgs)

if(NOT DEFINED Pthreads_EXCEPTION_MODE)
  set(Pthreads_EXCEPTION_MODE "C")
else()
  if(NOT Pthreads_EXCEPTION_MODE STREQUAL "C" AND
     NOT Pthreads_EXCEPTION_MODE STREQUAL "CE" AND
     NOT Pthreads_EXCEPTION_MODE STREQUAL "SE")
    message(FATAL_ERROR "Only C, CE, and SE exception modes are allowed")
  endif()
  if(NOT MSVC AND Pthreads_EXCEPTION_MODE STREQUAL "SE")
    message(FATAL_ERROR "Structured Exception Handling is only allowed for MSVC")
  endif()
endif()

find_path(Pthreads_INCLUDE_DIR
  NAMES pthread.h
  HINTS
    /usr/include
    /usr/local/include
)

if(MSVC)
  set(libnames pthreadV${Pthreads_EXCEPTION_SCHEME}2 pthread)
else()
  set(libnames pthread)
endif()

find_library(Pthreads_LIBRARY
  NAMES ${libnames}
  PATHS
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    /lib64
    /lib
)

find_package_handle_standard_args(Pthreads DEFAULT_MSG
  Pthreads_INCLUDE_DIR
  Pthreads_LIBRARY
)
