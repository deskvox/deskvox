include_directories(${VIRVO_INCLUDE_DIRS})

if(DESKVOX_BUILD_TESTS)
  add_subdirectory(test)
endif()
if(DESKVOX_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
add_subdirectory(virvo)

add_subdirectory(virvo/rayrend)

