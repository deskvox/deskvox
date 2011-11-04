# no default install path exists for protokit, so no standard paths given in here

IF(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)

  # Already in cache
  set (PROTOKIT_FOUND TRUE)

ELSE(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)

  FIND_PATH(PROTOKIT_INCLUDE_DIR protoApp.h
    DOC "The directory where protokit.h resides"
    )

  FIND_LIBRARY(PROTOKIT_LIBRARY Protokit
    DOC "The Protokit library"
    )

  IF(APPLE)
    SET(EXTRA_LIB "-lresolv")
  ENDIF(APPLE)
  IF(WIN32)
    SET(EXTRA_LIB ws2_32.lib iphlpapi.lib)
  ENDIF(WIN32)
    
  SET(PROTOKIT_LIBRARIES ${PROTOKIT_LIBRARY} ${EXTRA_LIB} CACHE STRING "The Protokit libraries")

  INCLUDE(FindPackageHandleStandardArgs)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(Protokit DEFAULT_MSG PROTOKIT_LIBRARIES PROTOKIT_INCLUDE_DIR)

ENDIF(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)
