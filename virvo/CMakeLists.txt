include_directories(
  ${VIRVO_CONFIG_DIR}
  ${VIRVO_INCLUDE_DIR}
  ${VIRVO_INCLUDE_DIR}/virvo
)

if(DESKVOX_BUILD_TESTS)
  add_subdirectory(test)
endif()
if(DESKVOX_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
add_subdirectory(virvo)
