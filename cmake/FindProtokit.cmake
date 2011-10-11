# no default install path exists for protokit, so no standard paths given in here

if (PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)

  # Already in cache
  set (PROTOKIT_FOUND TRUE)

else (PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)

  FIND_PATH(PROTOKIT_INCLUDE_DIR protoApp.h
	DOC "The directory where protokit.h resides"
	)

  FIND_LIBRARY(PROTOKIT_LIBRARIES Protokit
	DOC "The Protokit library"
	)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(PROTOKIT DEFAULT_MSG PROTOKIT_LIBRARIES PROTOKIT_INCLUDE_DIR)

endif (PROTOKIT_INCLUDE_DIR AND PROTOKIT_LIBRARIES)
