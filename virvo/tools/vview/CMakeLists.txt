set(SAVED_CMAKE_FIND_FRAMEWORK ${CMAKE_FIND_FRAMEWORK})
if(${VIRVO_WINDOWING_SYSTEM} STREQUAL "Cocoa")
    set(CMAKE_FIND_FRAMEWORK ONLY)
    find_package(GLUT)
else()
    set(CMAKE_FIND_FRAMEWORK NEVER)
    find_package(GLUT)
endif()
set(CMAKE_FIND_FRAMEWORK ${SAVED_CMAKE_FIND_FRAMEWORK})
if(NOT GLUT_FOUND)
  message(STATUS "Note: not building vview (GLUT not found)")
  return()
endif()

find_package(Boost REQUIRED)
#find_package(GLUT_REQUIRED)

deskvox_use_package(Boost)
deskvox_use_package(GLUT)

deskvox_link_libraries(virvo)
deskvox_link_libraries(virvo_fileio)

deskvox_add_tool(vview
  vvobjview.cpp
  vvobjview.h
  vvperformancetest.cpp
  vvperformancetest.h
  vvview.cpp
  vvview.h
)
