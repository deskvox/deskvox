include_directories(
  ${VIRVO_INCLUDE_DIRS}
  ${VOXDESK_INCLUDE_DIR}
)

include(Qt4-5)
find_qt()
if(NOT QT_FOUND)
    return()
endif()

if(DESKVOX_USE_ASIO)
  add_definitions(-DDESKVOX_USE_ASIO)
endif()

add_subdirectory(src)
add_subdirectory(plugins)
