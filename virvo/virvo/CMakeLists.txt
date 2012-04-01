CMAKE_MINIMUM_REQUIRED(VERSION 2.6.4)

IF(COVISE_BUILD)
  IF(NOT COVISE_USE_CUDA)
    SET(CUDA_FOUND NO)
  ENDIF(NOT COVISE_USE_CUDA) 
  COVISE_FIND_PACKAGE(PTHREAD REQUIRED)
ELSE(COVISE_BUILD)
  FIND_PACKAGE(PTHREAD REQUIRED)
ENDIF(COVISE_BUILD)

IF(WIN32)
  ADD_DEFINITIONS(-DUNICODE -DNOMINMAX -DVIRVO_EXPORT)
  IF(MSVC)
    IF(COVISE_BUILD)
      # TODO: no externlibs for non-covise build
      FIND_PATH(PTHREADW32_INCLUDE_DIR pthread.h PATHS "$ENV{EXTERNLIBS}/pthreads/include")
      FIND_LIBRARY(PTHREADW32_LIBRARY pthreadVC2.lib PATHS "$ENV{EXTERNLIBS}/pthreads/lib")
    ENDIF(COVISE_BUILD)
  ENDIF(MSVC)
ENDIF(WIN32)

IF(COVISE_BUILD)
  ADD_DEFINITIONS(-DNO_CONFIG_H -DVV_COVISE)
ELSE(COVISE_BUILD)
  INCLUDE(GenerateVersionInfoHeader)
ENDIF(COVISE_BUILD)

# Mandatory include dirs
INCLUDE_DIRECTORIES(
  ${OPENGL_INCLUDE_DIR}
  ${PTHREAD_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
)

# Optional include dirs
IF(CG_FOUND)
  INCLUDE_DIRECTORIES(${CG_INCLUDE_PATH})
ENDIF(CG_FOUND)

IF(NORM_FOUND)
  INCLUDE_DIRECTORIES(${NORM_INCLUDE_DIR})
ENDIF(NORM_FOUND)

SET(VIRVO_HEADERS
  vvbonjour/vvbonjour.h
  vvbonjour/vvbonjourbrowser.h
  vvbonjour/vvbonjourentry.h
  vvbonjour/vvbonjoureventloop.h
  vvbonjour/vvbonjourregistrar.h
  vvbonjour/vvbonjourresolver.h
  vvmultirend/vvtexmultirend.h
  vvmultirend/vvtexmultirendmngr.h
  vvaabb.h
  vvbrick.h
  vvbsptree.h
  vvbsptreevisitors.h
  vvcgprogram.h
  #vvclusterclient.h
  #vvclusterserver.h
  vvclock.h
  vvcomparisonrend.h
  vvcomparisonrend.impl.h
  vvcolor.h
  vvcuda.h
  vvcudaimg.h
  vvcudasw.h
  vvcudatools.h
  vvcudatransfunc.h
  vvcudautils.h
  vvdebugmsg.h
  vvdicom.h
  vvdynlib.h
  vvexport.h
  vvfileio.h
  vvglew.h
  vvglslprogram.h
  vvgltools.h
  vvibr.h
  vvibrclient.h
  vvibrimage.h
  vvibrserver.h
  vvvideo.h
  vvimage.h
  vvimageclient.h
  vvimageserver.h
  vvinttypes.h
  vvmulticast.h
  vvoffscreenbuffer.h
  vvopengl.h
  vvplatform.h
  vvprintgl.h
  vvpthread.h
  vvrayrend.h
  vvremoteclient.h
  vvremoteserver.h
  vvrendercontext.h
  vvrenderer.h
  vvrendererfactory.h
  vvrendertarget.h
  vvrendervp.h
  vvshaderfactory.h
  vvshaderprogram.h
  vvswitchrenderer.h
  vvswitchrenderer.impl.h
  vvsllist.h
  vvsocket.h
  vvsocketio.h
  vvsocketmonitor.h
  vvsoftimg.h
  vvsoftpar.h
  vvsoftper.h
  vvsoftvr.h
  vvsoftsw.h
  vvsphere.h
  vvstingray.h
  vvtcpserver.h
  vvtcpsocket.h
  vvtexrend.h
  vvtfwidget.h
  vvtokenizer.h
  vvtoolshed.h
  vvtransfunc.h
  vvudpsocket.h
  vvvecmath.h
  vvvecmath.impl.h
  vvvffile.h
  vvvirvo.h
  vvvisitor.h
  vvvoldesc.h
  vvx11.h 
)

SET(VIRVO_SOURCES
  vvbonjour/vvbonjour.cpp
  vvbonjour/vvbonjourbrowser.cpp
  vvbonjour/vvbonjourentry.cpp
  vvbonjour/vvbonjoureventloop.cpp
  vvbonjour/vvbonjourregistrar.cpp
  vvbonjour/vvbonjourresolver.cpp
  vvmultirend/vvtexmultirend.cpp
  vvmultirend/vvtexmultirendmngr.cpp
  vvaabb.cpp
  vvbrick.cpp
  vvbsptree.cpp
  vvbsptreevisitors.cpp
  vvcgprogram.cpp
  #vvclusterclient.cpp
  #vvclusterserver.cpp
  vvclock.cpp
  vvcolor.cpp
  vvcuda.cpp
  vvcudaimg.cpp
  vvcudatools.cpp
  vvdebugmsg.cpp
  vvdicom.cpp
  vvdynlib.cpp
  vvfileio.cpp
  vvglslprogram.cpp
  vvgltools.cpp
  vvibr.cpp
  vvibrclient.cpp
  vvibrimage.cpp
  vvibrserver.cpp
  vvvideo.cpp
  vvimage.cpp
  vvimageclient.cpp
  vvimageserver.cpp
  vvmulticast.cpp
  vvoffscreenbuffer.cpp
  vvprintgl.cpp
  vvpthread.cpp
  vvremoteclient.cpp
  vvremoteserver.cpp
  vvrendercontext.cpp
  vvrenderer.cpp
  vvrendererfactory.cpp
  vvrendertarget.cpp
  vvrendervp.cpp
  vvshaderfactory.cpp
  vvshaderprogram.cpp
  vvsocket.cpp
  vvsocketio.cpp
  vvsocketmonitor.cpp
  vvsoftimg.cpp
  vvsoftpar.cpp
  vvsoftper.cpp
  vvsoftvr.cpp
  vvsoftsw.cpp
  vvsphere.cpp
  vvstingray.cpp
  vvtcpserver.cpp
  vvtcpsocket.cpp
  vvtexrend.cpp
  vvtfwidget.cpp
  vvtokenizer.cpp
  vvtoolshed.cpp
  vvtransfunc.cpp
  vvudpsocket.cpp
  vvvecmath.cpp
  vvvffile.cpp
  vvvirvo.cpp
  vvvisitor.cpp
  vvvoldesc.cpp 
)

IF(${VIRVO_WINDOWING_SYSTEM})
  IF(APPLE)
    IF(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
      ADD_DEFINITIONS(-DUSE_COCOA)
      SET(VIRVO_HEADERS ${VIRVO_HEADERS} vvcocoaglcontext.h)
      SET(VIRVO_SOURCES ${VIRVO_SOURCES} vvcocoaglcontext.mm)
      FIND_LIBRARY(COCOA_LIBRARY Cocoa)
      SET(EXTRA_LIBS ${EXTRA_LIBS} ${COCOA_LIBRARY})
    ELSE(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
      IF(X11_FOUND)
        ADD_DEFINITIONS(-DUSE_X11)
      ENDIF(X11_FOUND)
    ENDIF(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
  ELSE(APPLE)
    IF(${VIRVO_WINDOWING_SYSTEM} STREQUAL "X11")
      IF(X11_FOUND)
        ADD_DEFINITION(-DUSE_X11)
      ENDIF(X11_FOUND)
    ENDIF(${VIRVO_WINDOWING_SYSTEM} STREQUAL "X11")
  ENDIF(APPLE)
ELSE(${VIRVO_WINDOWING_SYSTEM})
  IF(APPLE)
    ADD_DEFINITIONS(-DUSE_COCOA)
    SET(VIRVO_HEADERS ${VIRVO_HEADERS} vvcocoaglcontext.h)
    SET(VIRVO_SOURCES ${VIRVO_SOURCES} vvcocoaglcontext.mm)
    FIND_LIBRARY(COCOA_LIBRARY Cocoa)
    SET(EXTRA_LIBS ${EXTRA_LIBS} ${COCOA_LIBRARY})
  ELSE(APPLE)
    IF(X11_FOUND)
      ADD_DEFINITIONS(-DUSE_X11)
    ENDIF(X11_FOUND)
  ENDIF(APPLE)
ENDIF(${VIRVO_WINDOWING_SYSTEM})

IF(FFMPEG_FOUND)
  INCLUDE_DIRECTORIES(${FFMPEG_INCLUDE_PATHS})
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${FFMPEG_LIBRARIES})

  SET_SOURCE_FILES_PROPERTIES(vvvideo.cpp PROPERTIES COMPILE_FLAGS "${WNO_ERROR_FLAG}")
ENDIF(FFMPEG_FOUND)

IF(SNAPPY_FOUND)
  INCLUDE_DIRECTORIES(${SNAPPY_INCLUDE_DIR})
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${SNAPPY_LIBRARIES})
ENDIF(SNAPPY_FOUND)

IF(VOLPACK_FOUND)
  INCLUDE_DIRECTORIES(${VOLPACK_INCLUDE_DIR})
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${VOLPACK_LIBRARIES})
ENDIF(VOLPACK_FOUND)

IF(CUDA_FOUND)
  SET(VIRVO_CUDA_SOURCES
    vvcudasw.cu
    vvcudatransfunc.cu
    vvrayrend.cu
  )

  SET(VIRVO_SOURCES ${VIRVO_SOURCES} ${VIRVO_CUDA_SOURCES})
  SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
  INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
  IF(COVISE_BUILD)
    ADD_DEFINITIONS(-DHAVE_CUDA)
  ENDIF(COVISE_BUILD)
  CUDA_COMPILE(VIRVO_CUDA_OBJ ${VIRVO_CUDA_SOURCES} SHARED)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${CUDA_LIBRARIES})
ENDIF(CUDA_FOUND)

# Optional libraries
IF(CG_FOUND)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${CG_LIBRARY} ${CG_GL_LIBRARY})
ENDIF(CG_FOUND)

IF(BONJOUR_FOUND)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${BONJOUR_LIBRARIES})
ENDIF(BONJOUR_FOUND)

IF(NORM_FOUND)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ${NORM_LIBRARIES})
  SET_SOURCE_FILES_PROPERTIES(vvmulticast.cpp PROPERTIES COMPILE_FLAGS "${WNO_ERROR_FLAG}")
ENDIF(NORM_FOUND)

IF(APPLE)
  # hack to find glx on mac osx >= leopard
  IF(EXISTS ${CMAKE_OSX_SYSROOT}/usr/X11/include/GL/glx.h)
    SET(OPENGL_INCLUDE_DIR /usr/X11/include)
    SET(OPENGL_LIBRARIES /usr/X11/lib/libGL.dylib)
  ELSEIF(EXISTS ${CMAKE_OSX_SYSROOT}/usr/X11R6/include/GL/glx.h)
    SET(OPENGL_INCLUDE_DIR /usr/X11R6/include)
    SET(OPENGL_LIBRARIES /usr/X11R6/lib/libGL.dylib)
  ENDIF()
  INCLUDE_DIRECTORIES(BEFORE SYSTEM ${OPENGL_INCLUDE_DIR})
  SET(EXTRA_LIBS ${X11_X11_LIB} ${OPENGL_LIBRARIES} ${EXTRA_LIBS})
  SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-dylib_file,/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib")
ELSE(APPLE)
  IF(X11_FOUND)
    SET(EXTRA_LIBS ${EXTRA_LIBS} ${X11_LIBRARIES})
  ENDIF(X11_FOUND)
ENDIF(APPLE)

IF(WIN32)
  SET(EXTRA_LIBS ${EXTRA_LIBS} ws2_32.lib)
ENDIF(WIN32)

SET(VIRVO_EXTERNAL_LIBRARIES ${GLEW_LIBRARY} ${GLU_LIBRARY} ${OPENGL_LIBRARIES} ${PTHREAD_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${EXTRA_LIBS})

IF(COVISE_BUILD)
  ADD_COVISE_LIBRARY(coVirvo SHARED ${VIRVO_SOURCES} ${VIRVO_HEADERS} ${VIRVO_CUDA_OBJ})
  TARGET_LINK_LIBRARIES(coVirvo ${VIRVO_EXTERNAL_LIBRARIES})

  COVISE_WNOERROR(coVirvo)

  COVISE_INSTALL_TARGET(coVirvo)
ELSE(COVISE_BUILD)
  ADD_DEFINITIONS(-DNODLL) 
  ADD_LIBRARY(virvo STATIC ${VIRVO_HEADERS} ${VIRVO_CONFIG_DIR}/${VIRVO_CONFIG_H}
                                 ${VIRVO_VERSIONINFO_DIR}/${VIRVO_VERSIONINFO_H} ${VIRVO_SOURCES} ${VIRVO_CUDA_OBJ})
  TARGET_LINK_LIBRARIES(virvo ${VIRVO_EXTERNAL_LIBRARIES})

  INSTALL(FILES ${VIRVO_VERSIONINFO_H} 
    DESTINATION ${INSTALL_DIR_INCLUDE}/virvo
  )

  INSTALL(FILES ${VIRVO_CONFIG_DIR}/${VIRVO_CONFIG_H}
    DESTINATION ${INSTALL_DIR_INCLUDE}/virvo
  )

  INSTALL(FILES ${VIRVO_HEADERS}
    DESTINATION ${INSTALL_DIR_INCLUDE}/virvo
  )

  INSTALL(TARGETS virvo
    ARCHIVE DESTINATION ${INSTALL_DIR_LIB}
  )
ENDIF(COVISE_BUILD)
