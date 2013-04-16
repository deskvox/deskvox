if(NOT DESKVOX_BUILD_TESTS)
  return()
endif()

if(NOT BUILD_SHARED_LIBS)
  add_definitions(-DVIRVO_STATIC)
endif()

deskvox_link_libraries(virvo)

add_subdirectory(vvbonjour)
add_subdirectory(vvmulticast)
add_subdirectory(vvstopwatch)
