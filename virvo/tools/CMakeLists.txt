if(NOT DESKVOX_BUILD_TOOLS)
  return()
endif()

add_definitions(-DVV_APPLICATION_BUILD)

deskvox_link_libraries(virvo)

add_subdirectory(vconv)
add_subdirectory(vserver)
add_subdirectory(vview)
