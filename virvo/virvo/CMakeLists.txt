# @file
# 
# CMakeLists.txt for kernel - virvo
#
# @author Blasius Czink

IF(WIN32)
  ADD_DEFINITIONS(-DVIRVO_EXPORT -DUNICODE -DNOMINMAX)
  FIND_PATH(PTHREADW32_INCLUDE_DIR pthread.h PATHS "$ENV{EXTERNLIBS}/pthreads/include")
  FIND_LIBRARY(PTHREADW32_LIBRARY pthreadVC2.lib PATHS "$ENV{EXTERNLIBS}/pthreads/lib")
ENDIF(WIN32)

ADD_DEFINITIONS(-DNO_CONFIG_H -DVV_COVISE)

INCLUDE_DIRECTORIES(
  ${OPENGL_INCLUDE_DIR}
  ${PTHREADW32_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CG_INCLUDE_DIR}
  vvbonjour
  vvmultirend
)

SET(VIRVO_SOURCES
  vvaabb.cpp
  vvbrick.cpp
  vvcolor.cpp
  vvdebugmsg.cpp
  vvclusterclient.cpp
  vvclusterserver.cpp
  vvcuda.cpp
  vvcudaimg.cpp
  vvdicom.cpp
  vvdynlib.cpp
  vvfileio.cpp
  vvgltools.cpp
  vvideo.cpp
  vvimage.cpp
  vvoffscreenbuffer.cpp
  vvprintgl.cpp
  vvpthread.cpp
  vvremoteclient.cpp
  vvrendercontext.cpp
  vvrenderer.cpp
  vvrendertarget.cpp
  vvsocket.cpp
  vvsocketio.cpp
  vvsocketmonitor.cpp
  vvsphere.cpp
  vvstingray.cpp
  vvstopwatch.cpp
  vvtexrend.cpp
  vvtfwidget.cpp
  vvtokenizer.cpp
  vvtoolshed.cpp
  vvtransfunc.cpp
  vvvecmath.cpp
  vvvisitor.cpp
  vvvffile.cpp
  vvvirvo.cpp
  vvvoldesc.cpp
  vvshaderfactory.cpp
  vvshadermanager.cpp
  vvcg.cpp
  vvglsl.cpp
  vvbsptree.cpp
  vvbsptreevisitors.cpp
  vvsoftimg.cpp
  vvsoftpar.cpp
  vvsoftper.cpp
  vvsoftvr.cpp
  vvbonjour/vvbonjourbrowser.cpp
  vvbonjour/vvbonjourentry.cpp
  vvbonjour/vvbonjourregistrar.cpp
  vvbonjour/vvbonjourresolver.cpp
  vvmultirend/vvtexmultirend.cpp
  vvmultirend/vvtexmultirendmngr.cpp
  vvibrclient.cpp
  vvibrserver.cpp
  vvremoteserver.cpp
)

SET(VIRVO_HEADERS
  vvaabb.h
  vvarray.h
  vvbrick.h
  vvclusterclient.h
  vvclusterserver.h
  vvcolor.h
  vvcudatransfunc.h
  vvdebugmsg.h
  vvcuda.h
  vvcudaimg.h
  vvcudautils.h
  vvdicom.h
  vvdynlib.h
  vvexport.h
  vvfileio.h
  vvgltools.h
  vvideo.h
  vvimage.h
  vvoffscreenbuffer.h
  vvopengl.h
  vvglew.h
  vvprintgl.h
  vvpthread.h
  vvrayrend.h
  vvremoteclient.h
  vvrendercontext.h
  vvrenderer.h
  vvrendertarget.h
  vvsllist.h
  vvsocket.h
  vvsocketio.h
  vvsocketmonitor.h
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
  vvshaderfactory.h
  vvshadermanager.h
  vvcg.h
  vvglsl.h
  vvbsptree.h
  vvx11.h
  vvbsptreevisitors.h
  vvsoftimg.h
  vvsoftpar.h
  vvsoftper.h
  vvsoftvr.h
  vvbonjour/vvbonjourbrowser.h
  vvbonjour/vvbonjourentry.h
  vvbonjour/vvbonjourregistrar.h
  vvbonjour/vvbonjourresolver.h
  vvmultirend/vvtexmultirend.h
  vvmultirend/vvtexmultirendmngr.h
  vvibrclient.h
  vvibrserver.h
  vvremoteserver.h
)

IF(CUDA_FOUND)
  SET(VIRVO_CUDA_SOURCES vvcudatransfunc.cu vvcudasw.cu vvrayrend.cu)
  ADD_DEFINITIONS(-DHAVE_CUDA -DNV_PROPRIETARY_CODE)
  SET(VIRVO_CUDA_TGT VirvoCudaLib)
  CUDA_ADD_LIBRARY(${VIRVO_CUDA_TGT} ${VIRVO_CUDA_SOURCES} STATIC)
ENDIF(CUDA_FOUND)

IF(WIN32)
  SET(EXTRA_LIBS ${PTHREADW32_LIBRARY} ws2_32.lib)
ENDIF(WIN32)

ADD_COVISE_LIBRARY(coVirvo SHARED ${VIRVO_SOURCES} ${VIRVO_HEADERS})
TARGET_LINK_LIBRARIES(coVirvo ${VIRVO_CUDA_TGT} ${CG_LIBRARY} ${CG_GL_LIBRARY} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${EXTRA_LIBS})

COVISE_WNOERROR(coVirvo)

COVISE_INSTALL_TARGET(coVirvo)
