# - Try to find FFMPEG
# Once done this will define
#  
#  FFMPEG_FOUND		 - system has FFMPEG
#  FFMPEG_INCLUDE_DIR	 - the include directories
#  FFMPEG_LIBRARY_DIR	 - the directory containing the libraries
#  FFMPEG_LIBRARIES	 - link these to use FFMPEG
#  FFMPEG_SWSCALE_FOUND	 - FFMPEG also has SWSCALE
#   

SET( FFMPEG_HEADERS avformat.h avcodec.h avutil.h )
SET( FFMPEG_PATH_SUFFIXES libavformat libavcodec libavutil )
SET( FFMPEG_SWS_HEADERS swscale.h )
SET( FFMPEG_SWS_PATH_SUFFIXES libswscale )

if( WIN32 )
   SET( FFMPEG_LIBRARY_DIR $ENV{FFMPEGDIR}\\lib $ENV{EXTERNLIBS}\\ffmpeg\\lib )
   SET( FFMPEG_INCLUDE_PATHS $ENV{FFMPEGDIR}\\include $ENV{EXTERNLIBS}\\ffmpeg\\include)
   
  FIND_LIBRARY(FFMPEG_SWSCALE_LIBRARY NAMES swscale-0 swscale
  HINTS
     $ENV{FFMPEGDIR}
  PATH_SUFFIXES lib64 lib
  PATHS
    $ENV{EXTERNLIBS}/ffmpeg
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
    DOC "FFMPEG SWSCALE - Library"
  )
  FIND_LIBRARY(FFMPEG_AVFORMAT_LIBRARY NAMES avformat-52 avformat
  HINTS
     $ENV{FFMPEGDIR}
  PATH_SUFFIXES lib64 lib
  PATHS
    $ENV{EXTERNLIBS}/ffmpeg
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
    DOC "FFMPEG AVFORMAT - Library"
  )
  
  FIND_LIBRARY(FFMPEG_AVUTIL_LIBRARY NAMES avutil-49 avutil
  HINTS
     $ENV{FFMPEGDIR}
  PATH_SUFFIXES lib64 lib
  PATHS
    $ENV{EXTERNLIBS}/ffmpeg
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
    DOC "FFMPEG AVUTIL - Library"
  )
  
  FIND_LIBRARY(FFMPEG_CODEC_LIBRARY NAMES avcodec-52 avcodec-51 avcodec
  HINTS
     $ENV{FFMPEGDIR}
  PATH_SUFFIXES lib64 lib
  PATHS
    $ENV{EXTERNLIBS}/ffmpeg
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
    DOC "FFMPEG AVCODEC - Library"
  )
   SET( FFMPEG_SWS_LIBRARIES ${FFMPEG_SWSCALE_LIBRARY} )
   SET( FFMPEG_LIBRARIES ${FFMPEG_AVFORMAT_LIBRARY} ${FFMPEG_CODEC_LIBRARY} ${FFMPEG_AVUTIL_LIBRARY} )
   # check to see if we can find swscale
   
   IF ( FFMPEG_SWSCALE_LIBRARY )
      SET( SWSCALE_FOUND TRUE )
   ENDIF( FFMPEG_SWSCALE_LIBRARY )
else( WIN32 )
   SET(ENV{PKG_CONFIG_PATH} "$ENV{EXTERNLIBS}/ffmpeg/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
   SET( FFMPEG_LIBRARIES avformat avcodec avutil )
   SET( FFMPEG_SWS_LIBRARIES swscale )
   INCLUDE(FindPkgConfig)
   if ( PKG_CONFIG_FOUND )
      pkg_check_modules( AVFORMAT libavformat>=52 )
      pkg_check_modules( AVCODEC libavcodec )
      pkg_check_modules( AVUTIL libavutil )
      pkg_check_modules( SWSCALE libswscale )
      IF ( NOT (AVFORMAT_FOUND AND AVCODEC_FOUND AND AVUTIL_FOUND))
        RETURN()
      ENDIF ( NOT (AVFORMAT_FOUND AND AVCODEC_FOUND AND AVUTIL_FOUND))
   endif ( PKG_CONFIG_FOUND )
   SET( FFMPEG_LIBRARY_DIR   ${AVFORMAT_LIBRARY_DIRS}
			     ${AVCODEC_LIBRARY_DIRS}
			     ${AVUTIL_LIBRARY_DIRS} )
   SET( FFMPEG_INCLUDE_PATHS ${AVFORMAT_INCLUDE_DIRS}
			     ${AVCODEC_INCLUDE_DIRS}
			     ${AVUTIL_INCLUDE_DIRS} )
                           
   FIND_PATH(AVF
        NAMES avformat.h
        PATHS ${AVFORMAT_INCLUDE_DIRS}/libavformat /usr/include/libavformat)

   IF (AVF)
        ADD_DEFINITIONS("-DHAVE_FFMPEG_SEPARATE_INCLUDES")
   ENDIF (AVF)
endif( WIN32 )

# add in swscale if found
IF ( SWSCALE_FOUND )
   SET( FFMPEG_LIBRARY_DIR   ${FFMPEG_LIBRARY_DIR}
     			     ${SWSCALE_LIBRARY_DIRS} )
   SET( FFMPEG_INCLUDE_PATHS ${FFMPEG_INCLUDE_PATHS}
     			     ${SWSCALE_INCLUDE_DIRS} )
   SET( FFMPEG_HEADERS	     ${FFMPEG_HEADERS}
     			     ${FFMPEG_SWS_HEADERS} )
   SET( FFMPEG_PATH_SUFFIXES ${FFMPEG_PATH_SUFFIXES}
     			     ${FFMPEG_SWS_PATH_SUFFIXES} )
   SET( FFMPEG_LIBRARIES     ${FFMPEG_LIBRARIES}
     			     ${FFMPEG_SWS_LIBRARIES} )
ENDIF ( SWSCALE_FOUND )

# find includes
SET( INC_SUCCESS 0 )
SET( TMP_ TMP-NOTFOUND )
SET( FFMPEG_INCLUDE_DIR ${FFMPEG_INCLUDE_PATHS} )
FOREACH( INC_ ${FFMPEG_HEADERS} )

   FIND_PATH( TMP_ ${INC_}
	      PATHS ${FFMPEG_INCLUDE_PATHS}
	      PATH_SUFFIXES ${FFMPEG_PATH_SUFFIXES} )
   IF ( TMP_ )
      MATH( EXPR INC_SUCCESS ${INC_SUCCESS}+1 )
      SET( FFMPEG_INCLUDE_DIR ${FFMPEG_INCLUDE_DIR} ${TMP_} )
   ENDIF ( TMP_ )
   SET( TMP_ TMP-NOTFOUND )
ENDFOREACH( INC_ )

# find the full paths of the libraries
SET( TMP_ TMP-NOTFOUND )
IF ( NOT WIN32 )
   FOREACH( LIB_ ${FFMPEG_LIBRARIES} )
      FIND_LIBRARY( TMP_ NAMES ${LIB_} PATHS ${FFMPEG_LIBRARY_DIR} )
      IF ( TMP_ )
	 SET( FFMPEG_LIBRARIES_FULL ${FFMPEG_LIBRARIES_FULL} ${TMP_} )
      ENDIF ( TMP_ )
      SET( TMP_ TMP-NOTFOUND )
   ENDFOREACH( LIB_ )
   SET ( FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES_FULL} )
ENDIF( NOT WIN32 )

LIST( LENGTH FFMPEG_HEADERS LIST_SIZE_ )

SET( FFMPEG_FOUND FALSE )
SET( FFMPEG_SWSCALE_FOUND FALSE )
IF ( ${INC_SUCCESS} EQUAL ${LIST_SIZE_} )
   SET( FFMPEG_FOUND TRUE )
   SET( FFMPEG_SWSCALE_FOUND ${SWSCALE_FOUND} )
ENDIF ( ${INC_SUCCESS} EQUAL ${LIST_SIZE_} )
