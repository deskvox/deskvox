# no default install path exists for protokit, so no standard paths given in here

FIND_PATH(PROTOKIT_INCLUDE_DIR protoApp.h
  DOC "The directory where protokit.h resides"
)

FIND_LIBRARY(PROTOKIT_LIBRARIES Protokit
  DOC "The Protokit library"
)

IF(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)
  SET(PROTOKIT_FOUND 1 CACHE STRING "Set to 1 if PROTOKIT is found, 0 otherwise")
ELSE(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)
  SET(PROTOKIT_FOUND 0 CACHE STRING "Set to 1 if PROTOKIT is found, 0 otherwise")
  IF(NOT PROTOKIT_INCLUDE_DIR)
    MESSAGE("ERROR: Couldn't find Protokit include directory")
  ENDIF(NOT PROTOKIT_INCLUDE_DIR)
  IF(NOT PROTOKIT_LIBRARIES)
    MESSAGE("ERROR: Couldn't find Protokit library directory")
  ENDIF(NOT PROTOKIT_LIBRARIES)
ENDIF(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)

MARK_AS_ADVANCED(PROTOKIT_FOUND)

