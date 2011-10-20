# - Find JPEGTURBO
# Find the libjpeg-turbo includes and library
# This module defines
#  JPEGTURBO_INCLUDE_DIR, where to find jpeglib.h and turbojpeg.h, etc.
#  JPEGTURBO_LIBRARIES, the libraries needed to use libjpeg-turbo.
#  JPEGTURBO_FOUND, If false, do not try to use libjpeg-turbo.
# also defined, but not for general use are
#  JPEGTURBO_LIBRARY, where to find the libjpeg-turbo library.

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

FIND_PATH(JPEGTURBO_INCLUDE_DIR turbojpeg.h)

FIND_LIBRARY(TURBOJPEG_LIBRARY NAMES turbojpeg)
FIND_LIBRARY(JPEGTURBO_LIBRARY NAMES jpeg)

# handle the QUIETLY and REQUIRED arguments and set JPEGTURBO_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(JPEGTURBO DEFAULT_MSG TURBOJPEG_LIBRARY JPEGTURBO_LIBRARY JPEGTURBO~

IF(JPEGTURBO_FOUND)
  SET(JPEGTURBO_LIBRARIES ${JPEGTURBO_LIBRARY})
  SET(TURBOJPEG_LIBRARIES ${TURBOJPEG_LIBRARY})
ENDIF(JPEGTURBO_FOUND)

MARK_AS_ADVANCED(TURBOJPEG_LIBRARY JPEGTURBO_LIBRARY JPEGTURBO_INCLUDE_DIR )
