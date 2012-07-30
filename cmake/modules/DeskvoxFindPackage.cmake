# FindDeskvoxPackages.cmake


#---------------------------------------------------------------------------------------------------
# __deskvox_add_package(name)
#


macro(__deskvox_add_package name)
  # Add include directory for the given package
  # Treat 3rd-party headers as system headers and ignore warnings...
  include_directories(SYSTEM ${DESKVOX_${name}_INCLUDE_DIR})

  # Link with libraries provided by the given package
  deskvox_link_libraries(${DESKVOX_${name}_LIBRARIES})
endmacro()


#---------------------------------------------------------------------------------------------------
# deskvox_use_package(name)
#


macro(deskvox_use_package name)
  if(NOT DESKVOX_${name}_FOUND)
    message(FATAL_ERROR ${name} " not found")
    return() # Return from function, file, ...
  endif()
  __deskvox_add_package(${name})
endmacro()


#---------------------------------------------------------------------------------------------------
# deskvox_use_optional_package(name)
#


macro(deskvox_use_optional_package name)
  if(NOT DESKVOX_${name}_FOUND)
    #message(STATUS "Note: " ${CMAKE_CURRENT_SOURCE_DIR} ": optional package " ${name} " not found")
  else()
    __deskvox_add_package(${name})
  endif()
endmacro()


#---------------------------------------------------------------------------------------------------
# find packages
#


find_package(Bonjour)
find_package(Boost)
find_package(Cg)
find_package(CUDA)
find_package(FFMPEG)
find_package(FOX)
find_package(GLEW)
find_package(GLUT)
find_package(NORM)
find_package(OpenGL)
find_package(Protokit)
find_package(Pthreads)
find_package(SNAPPY)
find_package(VolPack)
if(NOT WIN32)
    find_package(X11)
endif()


if(BONJOUR_FOUND)
  set(DESKVOX_Bonjour_FOUND 1)
  set(DESKVOX_Bonjour_INCLUDE_DIR ${BONJOUR_INCLUDE_DIR})
  set(DESKVOX_Bonjour_LIBRARIES ${BONJOUR_LIBRARIES})
else()
  set(DESKVOX_Bonjour_FOUND 0)
endif()

if(Boost_FOUND)
  set(DESKVOX_Boost_FOUND 1)
  set(DESKVOX_Boost_INCLUDE_DIR ${Boost_INCLUDE_DIR})
  set(DESKVOX_Boost_LIBRARIES ${Boost_LIBRARIES})
else()
  set(DESKVOX_Boost_FOUND 0)
endif()

if(CG_FOUND)
  set(DESKVOX_Cg_FOUND 1)
  set(DESKVOX_Cg_INCLUDE_DIR ${CG_INCLUDE_PATH})
  set(DESKVOX_Cg_LIBRARIES ${CG_LIBRARY} ${CG_GL_LIBRARY})
else()
  set(DESKVOX_Cg_FOUND 0)
endif()

if(CUDA_FOUND)
  set(DESKVOX_CUDA_FOUND 1)
  set(DESKVOX_CUDA_INCLUDE_DIR ${CUDA_TOOLKIT_INCLUDE})
  set(DESKVOX_CUDA_LIBRARIES ${CUDA_LIBRARIES})
else()
  set(DESKVOX_CUDA_FOUND 0)
endif()

if(FFMPEG_FOUND)
  set(DESKVOX_FFMPEG_FOUND 1)
  set(DESKVOX_FFMPEG_INCLUDE_DIR ${FFMPEG_INCLUDE_DIR})
  set(DESKVOX_FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES})
else()
  set(DESKVOX_FFMPEG_FOUND 0)
endif()

if(FOX_FOUND)
  set(DESKVOX_FOX_FOUND 1)
  set(DESKVOX_FOX_INCLUDE_DIR ${FOX_INCLUDE_DIR})
  set(DESKVOX_FOX_LIBRARIES ${FOX_LIBRARY})
else()
  set(DESKVOX_FOX_FOUND 0)
endif()

if(GLEW_FOUND)
  set(DESKVOX_GLEW_FOUND 1)
  set(DESKVOX_GLEW_INCLUDE_DIR ${GLEW_INCLUDE_DIR})
  set(DESKVOX_GLEW_LIBRARIES ${GLEW_LIBRARY})
else()
  set(DESKVOX_GLEW_FOUND 0)
endif()

if(GLUT_FOUND)
  set(DESKVOX_GLUT_FOUND 1)
  set(DESKVOX_GLUT_INCLUDE_DIR ${GLUT_INCLUDE_DIR})
  set(DESKVOX_GLUT_LIBRARIES ${GLUT_LIBRARY})
else()
  set(DESKVOX_GLUT_FOUND 0)
endif()

if(NORM_FOUND)
  set(DESKVOX_NORM_FOUND 1)
  set(DESKVOX_NORM_INCLUDE_DIR ${NORM_INCLUDE_DIRS})
  set(DESKVOX_NORM_LIBRARIES ${NORM_LIBRARIES})
else()
  set(DESKVOX_NORM_FOUND 0)
endif()

if(OPENGL_FOUND)
  set(DESKVOX_OpenGL_FOUND 1)
  set(DESKVOX_OpenGL_INCLUDE_DIR ${OPENGL_INCLUDE_DIR})
  set(DESKVOX_OpenGL_LIBRARIES ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
else()
  set(DESKVOX_OpenGL_FOUND 0)
endif()

if(PROTOKIT_FOUND)
  set(DESKVOX_Protokit_FOUND 1)
  set(DESKVOX_Protokit_INCLUDE_DIR ${PROTOKIT_INCLUDE_DIR})
  set(DESKVOX_Protokit_LIBRARIES ${PROTOKIT_LIBRARIES})
else()
  set(DESKVOX_Protokit_FOUND 0)
endif()

if(PTHREADS_FOUND)
  set(DESKVOX_Pthreads_FOUND 1)
  set(DESKVOX_Pthreads_INCLUDE_DIR ${Pthreads_INCLUDE_DIR})
  set(DESKVOX_Pthreads_LIBRARIES ${Pthreads_LIBRARY})
else()
  set(DESKVOX_Pthreads_FOUND 0)
endif()

if(SNAPPY_FOUND)
  set(DESKVOX_SNAPPY_FOUND 1)
  set(DESKVOX_SNAPPY_INCLUDE_DIR ${SNAPPY_INCLUDE_DIR})
  set(DESKVOX_SNAPPY_LIBRARIES ${SNAPPY_LIBRARIES})
else()
  set(DESKVOX_SNAPPY_FOUND 0)
endif()

if(VOLPACK_FOUND)
  set(DESKVOX_VolPack_FOUND 1)
  set(DESKVOX_VolPack_INCLUDE_DIR ${VOLPACK_INCLUDE_DIR})
  set(DESKVOX_VolPack_LIBRARIES ${VOLPACK_LIBRARIES})
else()
  set(DESKVOX_VolPack_FOUND 0)
endif()

if(X11_FOUND)
  set(DESKVOX_X11_FOUND 1)
  set(DESKVOX_X11_INCLUDE_DIR ${X11_INCLUDE_DIR})
  set(DESKVOX_X11_LIBRARIES ${X11_LIBRARIES})
else()
  set(DESKVOX_X11_FOUND 0)
endif()
