# no default install path exists for protokit, so no standard paths given in here

IF(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARY)

  # Already in cache
  set (PROTOKIT_FOUND TRUE)

ELSE(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARY)

  FIND_PATH(PROTOKIT_INCLUDE_DIR protoApp.h
    DOC "The directory where protokit.h resides"
    )

  FIND_LIBRARY(PROTOKIT_LIBRARY Protokit
    DOC "The Protokit library"
    )

  INCLUDE(FindPackageHandleStandardArgs)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(Protokit DEFAULT_MSG PROTOKIT_LIBRARY PROTOKIT_INCLUDE_DIR)

ENDIF(PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARY)
