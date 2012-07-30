if(NOT DESKVOX_GLUT_FOUND)
  message(STATUS "Note: not building vview (required package GLUT not found)")
  return()
endif()

deskvox_use_package(GLUT)

deskvox_add_tool(vview
  vvobjview.cpp
  vvobjview.h
  vvperformancetest.cpp
  vvperformancetest.h
  vvview.cpp
  vvview.h
)
