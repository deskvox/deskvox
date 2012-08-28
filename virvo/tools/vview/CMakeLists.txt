INCLUDE_DIRECTORIES(../vcommon)

#find_package(GLUT_REQUIRED)

find_package(GLUT)
if(NOT GLUT_FOUND)
  message(STATUS "Note: not building vview (GLUT not found)")
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
