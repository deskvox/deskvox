include(FindPackageHandleStandardArgs)

find_path(GLEW_INCLUDE_DIR
  NAMES GL/glew.h
  HINTS
    /usr/include
    /usr/local/include
)

find_library(GLEW_LIBRARY
  NAMES GLEW glew glew_static glew32 glew32_static
  PATHS
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
)

find_package_handle_standard_args(GLEW DEFAULT_MSG
  GLEW_INCLUDE_DIR
  GLEW_LIBRARY
)
