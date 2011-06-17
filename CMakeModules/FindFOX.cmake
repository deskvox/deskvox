FIND_PATH(FOX_INCLUDE_DIR fx.h
  /usr/include
  /usr/local/include/fox-1.6
  /usr/include/fox-1.6
  /opt/local/include/fox-1.6
  DOC "The directory where fx.h resides"
)

FIND_LIBRARY(FOX_LIBRARIES FOX-1.6
  /usr/lib
  /usr/lib32
  /usr/lib64
  /usr/local/lib
  /opt/local/lib
  DOC "The fox-1.6 library"
)

IF(FOX_INCLUDE_DIR AND FOX_LIBRARIES)
  SET(FOX_FOUND 1 CACHE STRING "Set to 1 if fox-1.6 is found, 0 otherwise")
ELSE(FOX_INCLUDE_DIR AND FOX_LIBRARIES)
  SET(FOX_FOUND 0 CACHE STRING "Set to 1 if fox-1.6 is found, 0 otherwise")
  IF(NOT FOX_INCLUDE_DIR)
    MESSAGE("ERROR: Couldn't find fox-1.6 include directory")
  ENDIF(NOT FOX_INCLUDE_DIR)
  IF(NOT FOX_LIBRARIES)
    MESSAGE("ERROR: Couldn't find fox-1.6 library directory")
  ENDIF(NOT FOX_LIBRARIES)
ENDIF(FOX_INCLUDE_DIR AND FOX_LIBRARIES)

MARK_AS_ADVANCED(FOX_FOUND)

