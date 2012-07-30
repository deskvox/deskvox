if(NOT DESKVOX_BUILD_TESTS)
  return()
endif()

add_definitions(-DVV_APPLICATION_BUILD)

deskvox_link_libraries(virvo)

add_subdirectory(vvbonjour)
add_subdirectory(vvmulticast)
add_subdirectory(vvstopwatch)
