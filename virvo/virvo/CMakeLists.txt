find_package(Bonjour)
find_package(Boost)
find_package(Cg)
find_package(CUDA)
find_package(FFMPEG)
find_package(GLEW REQUIRED)
find_package(NORM)
find_package(OpenGL REQUIRED)
find_package(Protokit)
find_package(Pthreads REQUIRED)
find_package(SNAPPY)
find_package(VolPack)
find_package(X11)

deskvox_use_package(Bonjour)
deskvox_use_package(Boost)
deskvox_use_package(Cg INCDIRS ${CG_INCLUDE_PATH} LIBS ${CG_LIBRARY} ${CG_GL_LIBRARY})
deskvox_use_package(CUDA INCDIRS ${CUDA_TOOLKIT_INCLUDE})
deskvox_use_package(FFMPEG)
deskvox_use_package(GLEW)
deskvox_use_package(NORM)
deskvox_use_package(OpenGL)
deskvox_use_package(Protokit)
deskvox_use_package(Pthreads)
deskvox_use_package(SNAPPY)
deskvox_use_package(VolPack)
deskvox_use_package(X11)


#---------------------------------------------------------------------------------------------------
# Create config file
#


foreach(p ${__DESKVOX_USED_PACKAGES})
  set(VV_HAVE_${p} 1)
endforeach()

configure_file(${VIRVO_INCLUDE_DIR}/virvo/vvconfig.h.in ${VIRVO_CONFIG_DIR}/vvconfig.h)


#---------------------------------------------------------------------------------------------------
# Add virvo library
#


add_definitions(-DHAVE_CONFIG_H)

add_definitions(-DVV_LIBRARY_BUILD)

if(NOT BUILD_SHARED_LIBS)
  add_definitions(-DVIRVO_STATIC)
else()
  add_definitions(-DVIRVO_EXPORTS)
endif()


set(VIRVO_HEADERS
  private/vvcompiler.h
  private/vvgltools.h
  private/vvlog.h
  private/vvvariadic.h

  cuda/array.h
  cuda/debug.h
  cuda/graphics_resource.h
  cuda/host_device_array.h
  cuda/memory.h
  cuda/symbol.h
  cuda/texture.h

  gl/handle.h

  vvbonjour/vvbonjour.h
  vvbonjour/vvbonjourbrowser.h
  vvbonjour/vvbonjourentry.h
  vvbonjour/vvbonjoureventloop.h
  vvbonjour/vvbonjourregistrar.h
  vvbonjour/vvbonjourresolver.h

  vvmultirend/vvtexmultirend.h
  vvmultirend/vvtexmultirendmngr.h

  vvaabb.h
  vvaabb.impl.h
  vvarray.h
  vvbrick.h
  vvbrickrend.h
  vvbsptree.h
  vvbsptreevisitors.h
  vvcgprogram.h
  vvclock.h
  vvcolor.h
  vvcuda.h
  vvcudaimg.h
  vvcudarendertarget.h
  vvcudatools.h
  vvcudatransfunc.h
  vvcudautils.h
  vvdebugmsg.h
  vvdicom.h
  vvdynlib.h
  vvexport.h
  vvfileio.h
  vvglslprogram.h
  vvibr.h
  vvibrclient.h
  vvibrimage.h
  vvibrrenderer.h
  vvibrserver.h
  vvimage.h
  vvimageclient.h
  vvimageserver.h
  vvinttypes.h
  vvmulticast.h
  vvoffscreenbuffer.h
  vvopengl.h
  vvparam.h
  vvparbrickrend.h
  vvplatform.h
  vvprintgl.h
  vvpthread.h
  vvrayrend-common.h
  vvrayrend.h
  vvrect.h
  vvremoteclient.h
  vvremoteevents.h
  vvremoteserver.h
  vvrendercontext.h
  vvrenderer.h
  vvrendererfactory.h
  vvrendertarget.h
  vvrendervp.h
  vvrequestmanagement.h
  vvserbrickrend.h
  vvshaderfactory.h
  vvshaderprogram.h
  vvsllist.h
  vvsocket.h
  vvsocketio.h
  vvsocketmap.h
  vvsocketmonitor.h
  vvsoftimg.h
  vvsoftpar.h
  vvsoftper.h
  vvsoftrayrend.h
  vvsoftsw.h
  vvsoftvr.h
  vvsphere.h
  vvstingray.h
  vvswitchrenderer.h
  vvswitchrenderer.impl.h
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
  vvvideo.h
  vvvirvo.h
  vvvisitor.h
  vvvoldesc.h
)

set(VIRVO_SOURCES
  private/vvgltools.cpp
  private/vvlog.cpp

  cuda/graphics_resource.cpp
  cuda/host_device_array.cpp

  gl/handle.cpp

  vvbonjour/vvbonjour.cpp
  vvbonjour/vvbonjourbrowser.cpp
  vvbonjour/vvbonjourentry.cpp
  vvbonjour/vvbonjoureventloop.cpp
  vvbonjour/vvbonjourregistrar.cpp
  vvbonjour/vvbonjourresolver.cpp

  vvmultirend/vvtexmultirend.cpp
  vvmultirend/vvtexmultirendmngr.cpp

  vvbrick.cpp
  vvbrickrend.cpp
  vvbsptree.cpp
  vvbsptreevisitors.cpp
  vvcgprogram.cpp
  vvclock.cpp
  vvcolor.cpp
  vvcuda.cpp
  vvcudaimg.cpp
  vvcudarendertarget.cpp
  vvcudatools.cpp
  vvcudatransfunc.cpp
  vvdebugmsg.cpp
  vvdicom.cpp
  vvdynlib.cpp
  vvfileio.cpp
  vvglslprogram.cpp
  vvibr.cpp
  vvibrclient.cpp
  vvibrimage.cpp
  vvibrrenderer.cpp
  vvibrserver.cpp
  vvimage.cpp
  vvimageclient.cpp
  vvimageserver.cpp
  vvmulticast.cpp
  vvoffscreenbuffer.cpp
  vvparbrickrend.cpp
  vvprintgl.cpp
  vvpthread.cpp
  vvrayrend.cpp
  vvremoteclient.cpp
  vvremoteserver.cpp
  vvrendercontext.cpp
  vvrenderer.cpp
  vvrendererfactory.cpp
  vvrendertarget.cpp
  vvrendervp.cpp
  vvrequestmanagement.cpp
  vvserbrickrend.cpp
  vvshaderfactory.cpp
  vvshaderprogram.cpp
  vvsocket.cpp
  vvsocketio.cpp
  vvsocketmap.cpp
  vvsocketmonitor.cpp
  vvsoftimg.cpp
  vvsoftpar.cpp
  vvsoftper.cpp
  vvsoftrayrend.cpp
  vvsoftsw.cpp
  vvsoftvr.cpp
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
  vvvideo.cpp
  vvvirvo.cpp
  vvvisitor.cpp
  vvvoldesc.cpp
)

deskvox_cuda_compiles(VIRVO_CUDA_SOURCES
  vvcudatransfunc.cu
  vvrayrend.cu
)

set(VIRVO_SHADERS
  ../shader/vv_ibr.fsh
  ../shader/vv_ibr.vsh
  ../shader/vv_isect_geom_only.gsh
  ../shader/vv_isect_geom_only.vsh
  ../shader/vv_isect_vert_geom_combined.gsh
  ../shader/vv_isect_vert_geom_combined.vsh
  ../shader/vv_isect_vert_only.vsh
  ../shader/vv_shader01.fsh
  ../shader/vv_shader02.fsh
  ../shader/vv_shader03.fsh
  ../shader/vv_shader04.fsh
  ../shader/vv_shader05.fsh
  ../shader/vv_shader06.fsh
  ../shader/vv_shader07.fsh
  ../shader/vv_shader08.fsh
  ../shader/vv_shader09.fsh
  ../shader/vv_shader10.fsh
  ../shader/vv_shader11.fsh
  ../shader/vv_shader12.fsh
  ../shader/vv_shader13.fsh
)

if(APPLE)
  set(VIRVO_WINDOWING_SYSTEM "Cocoa" CACHE STRING "Available windowing systems for render contexts (Cocoa | X11)")
elseif(WIN32)
  set(VIRVO_WINDOWING_SYSTEM "WGL" CACHE STRING "Available windowing systems for render contexts (WGL | X11)")
else()
  set(VIRVO_WINDOWING_SYSTEM "X11" CACHE STRING "Available windowing systems for render contexts (X11)")
endif()

if(APPLE)
  if(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
    add_definitions(-DUSE_COCOA)
    set(VIRVO_HEADERS ${VIRVO_HEADERS} vvcocoaglcontext.h)
    set(VIRVO_SOURCES ${VIRVO_SOURCES} vvcocoaglcontext.mm)
    find_library(COCOA_LIBRARY Cocoa)
    if(NOT COCOA_LIBRARY)
      message(FATAL_ERROR "Cocoa not found")
    endif()
    deskvox_link_libraries(${COCOA_LIBRARY})
  else(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
    if(X11_FOUND)
      ADD_DEFINITIONS(-DUSE_X11)
    endif(X11_FOUND)
  endif(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
else(APPLE)
  if(${VIRVO_WINDOWING_SYSTEM} STREQUAL "X11")
    if(X11_FOUND)
      add_definitions(-DUSE_X11)
    endif(X11_FOUND)
  endif(${VIRVO_WINDOWING_SYSTEM} STREQUAL "X11")
endif(APPLE)

if(WIN32)
  deskvox_link_libraries(ws2_32)
else()
  if(BUILD_SHARED_LIBS)
    deskvox_link_libraries(dl)
  endif()
endif()


deskvox_add_library(virvo
  ${VIRVO_HEADERS}
  ${VIRVO_SOURCES}
  ${VIRVO_CUDA_SOURCES}
  ${VIRVO_SHADERS}
)
