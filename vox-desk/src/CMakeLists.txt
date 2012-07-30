deskvox_use_package(OpenGL)
deskvox_use_package(FOX)

add_definitions(-DVV_APPLICATION_BUILD)

include_directories(
  ${VIRVO_CONFIG_DIR}
  ${VIRVO_INCLUDE_DIR}
  ${VIRVO_INCLUDE_DIR}/virvo
)

##
## TODO:
## public virvo headers should not include third-party library headers...
##

if(DESKVOX_CUDA_FOUND)
  include_directories(SYSTEM ${DESKVOX_CUDA_INCLUDE_DIR})
endif()

deskvox_link_libraries(virvo)

deskvox_add_tool(vox-desk
  vvartracking.cpp
  vvartracking.h
  vvcanvas.cpp
  vvcanvas.h
  vvclient.cpp
  vvclient.h
  vvdialogs.cpp
  vvdialogs.h
  vvmovie.cpp
  vvmovie.h
  vvobjview.cpp
  vvobjview.h
  vvprefwindow.cpp
  vvprefwindow.h
  vvshell.cpp
  vvshell.h
  vvsliceviewer.cpp
  vvsliceviewer.h
# vvtranslines.cpp
# vvtranslines.h
  vvtranswindow.cpp
  vvtranswindow.h
)
