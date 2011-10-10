# no default install path exists for norm, so no standard paths given in here

FIND_PATH(NORM_INCLUDE_DIR normApi.h
  DOC "The directory where normApi.h resides"
)

FIND_LIBRARY(NORM_LIBRARIES NORM-1.4b3
  DOC "The norm-1.4b3 library"
)

IF(NORM_INCLUDE_DIR AND NORM_LIBRARIES)
  SET(NORM_FOUND 1 CACHE STRING "Set to 1 if norm-1.4b3 is found, 0 otherwise")
ELSE(NORM_INCLUDE_DIR AND NORM_LIBRARIES)
  SET(NORM_FOUND 0 CACHE STRING "Set to 1 if norm-1.4b3 is found, 0 otherwise")

  IF(NOT NORM_INCLUDE_DIR)
    MESSAGE("ERROR: Couldn't find norm-1.4b3 include directory")
  ENDIF(NOT NORM_INCLUDE_DIR)

  IF(NOT NORM_LIBRARIES)
    MESSAGE("ERROR: Couldn't find norm-1.4b3 library")
  ENDIF(NOT NORM_LIBRARIES)

ENDIF(NORM_INCLUDE_DIR AND NORM_LIBRARIES)

MARK_AS_ADVANCED(NORM_FOUND)

