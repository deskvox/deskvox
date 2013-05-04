include(FindPackageHandleStandardArgs)

IF (WIN32)
  SET(GLUT_PACKAGE_DEFINITIONS "-DGLUT_NO_LIB_PRAGMA")
  FIND_PATH( GLUT_INCLUDE_DIR NAMES GL/glut.h 
    PATHS  
    $ENV{EXTERNLIBS}/glut/include
    ${GLUT_ROOT_PATH}/include )
  FIND_LIBRARY( GLUT_LIBRARY_DEBUG NAMES glutD glut32D freeglutD
    PATHS
    $ENV{EXTERNLIBS}/glut/lib
    ${OPENGL_LIBRARY_DIR}
    ${GLUT_ROOT_PATH}/Debug
    )
    FIND_LIBRARY( GLUT_LIBRARY_RELEASE NAMES glut glut32 freeglut
    PATHS
    ${OPENGL_LIBRARY_DIR}
    $ENV{EXTERNLIBS}/glut/lib
    ${GLUT_ROOT_PATH}/Release
    )
    IF(MSVC_IDE)
      IF (GLUT_LIBRARY_DEBUG AND GLUT_LIBRARY_RELEASE)
         SET(GLUT_LIBRARY optimized ${GLUT_LIBRARY_RELEASE} debug ${GLUT_LIBRARY_DEBUG})
      ELSE (GLUT_LIBRARY_DEBUG AND GLUT_LIBRARY_RELEASE)
         SET(GLUT_LIBRARY NOTFOUND)
         MESSAGE(STATUS "Could not find the debug AND release version of zlib")
      ENDIF (GLUT_LIBRARY_DEBUG AND GLUT_LIBRARY_RELEASE)
    ELSE(MSVC_IDE)
      STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_TOLOWER)
      IF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(GLUT_LIBRARY ${GLUT_LIBRARY_DEBUG})
      ELSE(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
         SET(GLUT_LIBRARY ${GLUT_LIBRARY_RELEASE})
      ENDIF(CMAKE_BUILD_TYPE_TOLOWER MATCHES debug)
    ENDIF(MSVC_IDE)
    MARK_AS_ADVANCED(GLUT_LIBRARY_DEBUG GLUT_LIBRARY_RELEASE)
ELSE (WIN32)

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

ENDIF (WIN32)

IF(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLUT DEFAULT_MSG GLUT_LIBRARY_RELEASE GLUT_LIBRARY_DEBUG GLUT_INCLUDE_DIR)
  MARK_AS_ADVANCED(GLUT_LIBRARY_RELEASE GLUT_LIBRARY_DEBUG)
    SET( GLUT_LIBRARIES ${GLUT_LIBRARY}) 
ELSE(MSVC)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLUT DEFAULT_MSG GLUT_LIBRARY GLUT_INCLUDE_DIR)
  MARK_AS_ADVANCED(GLUT_LIBRARY)
ENDIF(MSVC)
