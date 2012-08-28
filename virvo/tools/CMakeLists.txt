if(NOT DESKVOX_BUILD_TOOLS)
  return()
endif()

add_definitions(-DVV_APPLICATION_BUILD)
if(NOT BUILD_SHARED_LIBS)
  add_definitions(-DVIRVO_STATIC)
endif()

deskvox_link_libraries(virvo)

ADD_SUBDIRECTORY(vcommon)
ADD_SUBDIRECTORY(vconv)
ADD_SUBDIRECTORY(vserver)
ADD_SUBDIRECTORY(vview)
