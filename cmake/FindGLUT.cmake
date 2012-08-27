include(FindPackageHandleStandardArgs)

set(hints
  $ENV{LIB_BASE_PATH}/glut
  $ENV{LIB_BASE_PATH}/freeglut
)

set(paths
  /usr
  /usr/local
)

find_path(GLUT_INCLUDE_DIR
  NAMES
    GL/glut.h
    GL/freeglut.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
)

find_library(GLUT_LIBRARY
  NAMES
    GLUT # MacOS X Framework
    glut
    glut_static
    gluts
    glut32
    glut32_static
    glut32s
    freeglut
    freeglut_static
    freegluts
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

find_library(GLUT_LIBRARY_DEBUG
  NAMES
    glutd
    glutd_static
    glutsd
    glut32d
    glut32d_static
    glut32sd
    freeglutd
    freeglutd_static
    freeglutsd
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
)

if(GLUT_LIBRARY_DEBUG)
  set(GLUT_LIBRARIES optimized ${GLUT_LIBRARY} debug ${GLUT_LIBRARY_DEBUG})
else()
  set(GLUT_LIBRARIES ${GLUT_LIBRARY})
endif()

find_package_handle_standard_args(GLUT DEFAULT_MSG
  GLUT_INCLUDE_DIR
  GLUT_LIBRARY
)
