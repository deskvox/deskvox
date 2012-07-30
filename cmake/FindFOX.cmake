include(FindPackageHandleStandardArgs)

find_path(FOX_INCLUDE_DIR
  NAMES fx.h
  HINTS
    /usr/include
    /usr/include/fox-1.6
    /usr/local/include
    /usr/local/include/fox-1.6
)

find_library(FOX_LIBRARY
  NAMES FOX-1.6 fox-1.6
  PATHS
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
)

find_package_handle_standard_args(FOX DEFAULT_MSG
  FOX_INCLUDE_DIR
  FOX_LIBRARY
)
