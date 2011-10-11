# no default install path exists for norm, so no standard paths given in here

if (NORM_INCLUDE_DIR AND NORM_LIBRARIES)

	# Already in cache
	set (NORM_FOUND TRUE)

else (NORM_INCLUDE_DIR AND NORM_LIBRARIES)

	FIND_PATH(NORM_INCLUDE_DIR normApi.h
		DOC "The directory where normApi.h resides"
		)

	FIND_LIBRARY(NORM_LIBRARIES NORM-1.4b3

		DOC "The norm-1.4b3 library"
		)

	include(FindPackageHandleStandardArgs)
	find_package_handle_standard_args(NORM DEFAULT_MSG NORM_LIBRARIES NORM_INCLUDE_DIR)

endif (NORM_INCLUDE_DIR AND NORM_LIBRARIES)

