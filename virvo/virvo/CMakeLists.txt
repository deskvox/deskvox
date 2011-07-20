CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

IF(WIN32)
  ADD_DEFINITIONS(-DVIRVO_EXPORT -DUNICODE -DNOMINMAX)
  IF(MSVC)
    # TODO: no externlibs for non-covise build
    FIND_PATH(PTHREADW32_INCLUDE_DIR pthread.h PATHS "$ENV{EXTERNLIBS}/pthreads/include")
    FIND_LIBRARY(PTHREADW32_LIBRARY pthreadVC2.lib PATHS "$ENV{EXTERNLIBS}/pthreads/lib")
  ENDIF(MSVC)
ENDIF(WIN32)

IF(COVISE_BUILD)
  ADD_DEFINITIONS(-DNO_CONFIG_H -DVV_COVISE)

  INCLUDE_DIRECTORIES(
    ${OPENGL_INCLUDE_DIR}
    ${PTHREADW32_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${CG_INCLUDE_DIR}
    vvbonjour
    vvmultirend
  )
ELSE(COVISE_BUILD)
  INCLUDE(GenerateConfigHeader)
  INCLUDE(GenerateVersionInfoHeader)

  # Include dirs from extern libs
  IF(CG_FOUND)
    INCLUDE_DIRECTORIES(${CG_INCLUDE_PATH})
  ENDIF(CG_FOUND)

  IF(GLEW_FOUND)
    INCLUDE_DIRECTORIES(${GLEW_INCLUDE_PATH})
  ENDIF(GLEW_FOUND)
ENDIF(COVISE_BUILD)

SET(VIRVO_HEADERS
  vvaabb.h
  vvbrick.h
  vvbsptree.h
  vvbsptreevisitors.h
  vvcg.h
  vvclusterclient.h
  vvclusterserver.h
  vvcolor.h
  vvcuda.h
  vvcudaimg.h
  vvcudasw.h
  vvcudatransfunc.h
  vvcudautils.h
  vvdebugmsg.h
  vvdicom.h
  vvdynlib.h
  vvexport.h
  vvfileio.h
  vvglew.h
  vvglsl.h
  vvgltools.h
  vvibrclient.h
  vvibrserver.h
  vvideo.h
  vvimage.h
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
  vvrendertarget.h
  vvrendervp.h
  vvshaderfactory.h
  vvshadermanager.h
  vvsllist.h
  vvsocket.h
  vvsocketio.h
  vvsocketmonitor.h
  vvsoftimg.h
  vvsoftpar.h
  vvsoftper.h
  vvsoftvr.h
  vvsphere.h
  vvstingray.h
  vvstopwatch.h
  vvtexrend.h
  vvtfwidget.h
  vvtokenizer.h
  vvtoolshed.h
  vvtransfunc.h
  vvvecmath.h
  vvvffile.h
  vvvirvo.h
  vvvisitor.h
  vvvoldesc.h
  vvx11.h 
)
              
SET(VIRVO_SOURCES
  vvaabb.cpp
  vvbrick.cpp
  vvbsptree.cpp
  vvbsptreevisitors.cpp
  vvcg.cpp
  vvclusterclient.cpp
  vvclusterserver.cpp
  vvcolor.cpp
  vvcuda.cpp
  vvcudaimg.cpp
  vvdebugmsg.cpp
  vvdicom.cpp
  vvdynlib.cpp
  vvfileio.cpp
  vvglsl.cpp
  vvgltools.cpp
  vvibrclient.cpp
  vvibrserver.cpp
  vvideo.cpp
  vvimage.cpp
  vvoffscreenbuffer.cpp
  vvprintgl.cpp
  vvpthread.cpp
  vvremoteclient.cpp
  vvremoteserver.cpp
  vvrendercontext.cpp
  vvrenderer.cpp
  vvrendertarget.cpp
  vvshaderfactory.cpp
  vvshadermanager.cpp
  vvsocket.cpp
  vvsocketio.cpp
  vvsocketmonitor.cpp
  vvsoftimg.cpp
  vvsoftpar.cpp
  vvsoftper.cpp
  vvsoftvr.cpp
  vvsphere.cpp
  vvstingray.cpp
  vvstopwatch.cpp
  vvtexrend.cpp
  vvtfwidget.cpp
  vvtokenizer.cpp
  vvtoolshed.cpp
  vvtransfunc.cpp
  vvvecmath.cpp
  vvvffile.cpp
  vvvirvo.cpp
  vvvisitor.cpp
  vvvoldesc.cpp 
)

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
  SET(EXTRA_LIBS ${CUDA_LIBRARIES})
ENDIF(CUDA_FOUND)

IF(COVISE_BUILD)
  IF(WIN32)
    SET(EXTRA_LIBS ${EXTRA_LIBS} ${PTHREADW32_LIBRARY} ws2_32.lib)
  ENDIF(WIN32)

  ADD_COVISE_LIBRARY(coVirvo SHARED ${VIRVO_SOURCES} ${VIRVO_HEADERS} ${VIRVO_CUDA_OBJ})
  TARGET_LINK_LIBRARIES(coVirvo ${CG_LIBRARY} ${CG_GL_LIBRARY} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${EXTRA_LIBS})

  COVISE_WNOERROR(coVirvo)

  COVISE_INSTALL_TARGET(coVirvo)
ELSE(COVISE_BUILD)
  ADD_LIBRARY(virvo SHARED ${VIRVO_HEADERS} ${VIRVO_CONFIG_H} ${VIRVO_VERSIONINFO_H} ${VIRVO_SOURCES} ${VIRVO_CUDA_OBJ})
  TARGET_LINK_LIBRARIES(virvo ${EXTRA_LIBS})

  IF(CG_FOUND)
    TARGET_LINK_LIBRARIES(virvo ${CG_LIBRARY})
    TARGET_LINK_LIBRARIES(virvo ${CG_GL_LIBRARY})
  ENDIF(CG_FOUND)

  IF(OPENGL_FOUND)
    TARGET_LINK_LIBRARIES(virvo ${OPENGL_gl_LIBRARY})
  ENDIF(OPENGL_FOUND)

  IF(OPENGL_GLU_FOUND)
    TARGET_LINK_LIBRARIES(virvo ${OPENGL_glu_LIBRARY})
  ENDIF(OPENGL_GLU_FOUND)

  IF(GLEW_FOUND)
    TARGET_LINK_LIBRARIES(virvo ${GLEW_LIBRARY})
  ENDIF(GLEW_FOUND)

  IF(X11_FOUND)
    TARGET_LINK_LIBRARIES(virvo ${X11_LIBRARIES})
  ENDIF(X11_FOUND)

  IF(WIN32)
    # TODO: no libdl, pthreads don't come with OS
  ELSE(WIN32)
    TARGET_LINK_LIBRARIES(virvo pthread dl)
  ENDIF(WIN32)
ENDIF(COVISE_BUILD)
