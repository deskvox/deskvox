include_directories(
  ${VIRVO_INCLUDE_DIRS}
  ${VOXDESK_INCLUDE_DIR}
)

include(Qt4-5)

if(DESKVOX_USE_ASIO)
  add_definitions(-DDESKVOX_USE_ASIO)
endif()

add_subdirectory(src)
add_subdirectory(plugins)
