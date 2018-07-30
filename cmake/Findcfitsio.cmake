# FindCfitsio.cmake - Find cfitsio library.
#
# This module defines the following variables:
#
# CFITSIO_FOUND: TRUE iff cfitsio is found.
# CFITSIO_INCLUDE_DIRS: Include directories for cfitsio.
# CFITSIO_LIBRARIES: Libraries required to link cfitsio.
#
# The following variables control the behaviour of this module:
#
# CFITSIO_INCLUDE_DIR_HINTS: List of additional directories in which to
#                            search for cfitsio includes, e.g: /home/include.
# CFITSIO_LIBRARY_DIR_HINTS: List of additional directories in which to
#                            search for cfitsio libraries, e.g: /home/lib.
#
# The following variables are also defined by this module, but in line with
# CMake recommended FindPackage() module style should NOT be referenced directly
# by callers (use the plural variables detailed above instead).  These variables
# do however affect the behaviour of the module via FIND_[PATH/LIBRARY]() which
# are NOT re-called (i.e. search for library is not repeated) if these variables
# are set with valid values _in the CMake cache_. This means that if these
# variables are set directly in the cache, either by the user in the CMake GUI,
# or by the user passing -DVAR=VALUE directives to CMake when called (which
# explicitly defines a cache variable), then they will be used verbatim,
# bypassing the HINTS variables and other hard-coded search locations.
#
# CFITSIO_INCLUDE_DIR: Include directory for cfitsio, not including the
#                      include directory of any dependencies.
# CFITSIO_LIBRARY: cfitsio library, not including the libraries of any
#                  dependencies.

# Called if we failed to find cfitsio or any of it's required dependencies,
# unsets all public (designed to be used externally) variables and reports
# error message at priority depending upon [REQUIRED/QUIET/<NONE>] argument.
MACRO(CFITSIO_REPORT_NOT_FOUND REASON_MSG)
  UNSET(CFITSIO_FOUND)
  UNSET(CFITSIO_INCLUDE_DIRS)
  UNSET(CFITSIO_LIBRARIES)
  # Make results of search visible in the CMake GUI if cfitsio has not
  # been found so that user does not have to toggle to advanced view.
  MARK_AS_ADVANCED(CLEAR CFITSIO_INCLUDE_DIR
                         CFITSIO_LIBRARY)
  # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by FindPackage()
  # use the camelcase library name, not uppercase.
  IF (Cfitsio_FIND_QUIETLY)
    MESSAGE(STATUS "Failed to find cfitsio - " ${REASON_MSG} ${ARGN})
ELSEIF (Cfitsio_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Failed to find cfitsio - " ${REASON_MSG} ${ARGN})
  ELSE()
    # Neither QUIETLY nor REQUIRED, use no priority which emits a message
    # but continues configuration and allows generation.
    MESSAGE("-- Failed to find cfitsio - " ${REASON_MSG} ${ARGN})
  ENDIF ()
ENDMACRO(CFITSIO_REPORT_NOT_FOUND)

# Search user-installed locations first, so that we prefer user installs
# to system installs where both exist.
#
LIST(APPEND CFITSIO_CHECK_INCLUDE_DIRS
  /usr/local/include
  /usr/local/homebrew/include # Mac OS X
  /opt/local/var/macports/software # Mac OS X.
  /opt/local/include
  /usr/include)
LIST(APPEND CFITSIO_CHECK_LIBRARY_DIRS
  /usr/local/lib
  /usr/local/homebrew/lib # Mac OS X.
  /opt/local/lib
  /usr/lib)

# Search supplied hint directories first if supplied.
FIND_PATH(CFITSIO_INCLUDE_DIR
  NAMES fitsio.h
  PATHS ${CFITSIO_INCLUDE_DIR_HINTS}
  ${CFITSIO_CHECK_INCLUDE_DIRS})
IF (NOT CFITSIO_INCLUDE_DIR OR
    NOT EXISTS ${CFITSIO_INCLUDE_DIR})
  CFITSIO_REPORT_NOT_FOUND(
      "Could not find cfitsio include directory, set CFITSIO_INCLUDE_DIR "
      "to directory containing cfitsio.h")
ENDIF (NOT CFITSIO_INCLUDE_DIR OR
       NOT EXISTS ${CFITSIO_INCLUDE_DIR})

FIND_LIBRARY(CFITSIO_LIBRARY NAMES cfitsio
    PATHS ${CFITSIO_LIBRARY_DIR_HINTS}
  ${CFITSIO_CHECK_LIBRARY_DIRS})
IF (NOT CFITSIO_LIBRARY OR
    NOT EXISTS ${CFITSIO_LIBRARY})
  CFITSIO_REPORT_NOT_FOUND(
    "Could not find cfitsio library, set CFITSIO_LIBRARY "
    "to full path to libcfitsio.")
ENDIF (NOT CFITSIO_LIBRARY OR
    NOT EXISTS ${CFITSIO_LIBRARY})

# Mark internally as found.
SET(CFITSIO_FOUND TRUE)

# Set standard CMake FindPackage variables if found.
IF (CFITSIO_FOUND)
    SET(CFITSIO_INCLUDE_DIRS ${CFITSIO_INCLUDE_DIR})
    SET(CFITSIO_LIBRARIES ${CFITSIO_LIBRARY})
ENDIF (CFITSIO_FOUND)

# Handle REQUIRED / QUIET optional arguments.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Cfitsio DEFAULT_MSG
    CFITSIO_INCLUDE_DIRS CFITSIO_LIBRARIES)

# Only mark internal variables as advanced if we found cfitsio, otherwise
# leave them visible in the standard GUI for the user to set manually.
IF (CFITSIO_FOUND)
    MARK_AS_ADVANCED(FORCE CFITSIO_INCLUDE_DIR
                           CFITSIO_LIBRARY)
ENDIF (CFITSIO_FOUND)
