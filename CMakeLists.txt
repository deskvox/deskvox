project(Deskvox)


cmake_minimum_required(VERSION 2.8)

# Add path for custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
)


#set(DESKVOX_VERSION_MAJOR 1)
#set(DESKVOX_VERSION_MINOR 0)

#set(PACKAGE_NAME DESKVOX)
#set(PACKAGE_VERSION "${DESKVOX_VERSION_MAJOR}.${DESKVOX_VERSION_MINOR}")
#set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
#set(PACKAGE_BUGREPORT "http://sourceforge.net/tracker/?group_id=131783&atid=722235")


set(DESKVOX_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DESKVOX_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(DESKVOX_PREFIX ${CMAKE_INSTALL_PREFIX})


set(VIRVO_VERSION_MAJOR 1)
set(VIRVO_VERSION_MINOR 0)
set(VIRVO_VERSION "${VIRVO_VERSION_MAJOR}.${VIRVO_VERSION_MINOR}")

set(VIRVO_DIR ${DESKVOX_DIR}/virvo)
set(VIRVO_INCLUDE_DIR ${VIRVO_DIR})
set(VIRVO_SOURCE_DIR ${VIRVO_DIR}/virvo)
set(VIRVO_SHADER_DIR ${VIRVO_DIR}/shader)
set(VIRVO_TOOLS_DIR ${VIRVO_DIR}/tools)
set(VIRVO_TESTS_DIR ${VIRVO_DIR}/test)

set(VIRVO_CONFIG_DIR ${DESKVOX_BINARY_DIR}/virvo/virvo/config)

set(VIRVO_INCLUDE_DIRS
  ${VIRVO_CONFIG_DIR}
  ${VIRVO_INCLUDE_DIR}
  ${VIRVO_INCLUDE_DIR}/virvo
)

set(VOXDESK_DIR ${DESKVOX_DIR}/vox-desk-qt)
set(VOXDESK_INCLUDE_DIR ${VOXDESK_DIR}/include)
set(VOXDESK_SOURCE_DIR ${VOXDESK_DIR}/src)
set(VOXDESK_PLUGINS_DIR ${VOXDESK_DIR}/plugins)


if(NOT COVISE_BUILD)
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${DESKVOX_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin")
   set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${DESKVOX_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin")
   set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${DESKVOX_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib")
endif(NOT COVISE_BUILD)


#---------------------------------------------------------------------------------------------------
# Options
#


# Override the default value specified in FindCUDA.cmake
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF CACHE BOOL "Attach the build rule to the CUDA source file.")


option(BUILD_SHARED_LIBS "Build all libraries as shared libraries instead of static" ON)

option(DESKVOX_USE_FOLDERS "Enable solution folders in Visual Studio. Disable for Express versions." ON)

option(DESKVOX_GROUP_SOURCES_BY_TYPE "Group source files by type." ON)

option(DESKVOX_BUILD_TESTS "Build tests" OFF)
option(DESKVOX_BUILD_TOOLS "Build tools" ON)
if (NOT DEFINED DESKVOX_BUILD_VOX_DESK)
  # Make this an option only if the variable has not otherwise been set. Allows
  # other CMake modules to override the default value defined here.
  option(DESKVOX_BUILD_VOX_DESK "Build vox-desk. Disable if Qt 4 is not available." ON)
endif()
option(DESKVOX_BUILD_VOX_DESK_FOX "Build fox-based vox-desk." OFF)

option(DESKVOX_ENABLE_WARNINGS "Enable all warnings" ON)
if(MSVC)
  option(DESKVOX_ENABLE_PEDANTIC "Compile with pedantic enabled (Ignored if warnings are disabled)" OFF)
else()
  option(DESKVOX_ENABLE_PEDANTIC "Compile with pedantic enabled (Ignored if warnings are disabled)" ON)
endif()

option(DESKVOX_ENABLE_WERROR "Fail and stop if a warning is triggered" OFF)

if (NOT DEFINED DESKVOX_USE_CUDA)
  # Make this an option only if the variable has not otherwise been set. Allows
  # other CMake modules to override the default value defined here.
  if (MINGW)
    option(DESKVOX_USE_CUDA "Use CUDA if available" OFF)
  else()
    option(DESKVOX_USE_CUDA "Use CUDA if available" ON)
  endif()
endif()

if (NOT DEFINED DESKVOX_USE_QT4)
  # Make this an option only if the variable has not otherwise been set. Allows
  # other CMake modules to override the default value defined here.
  option(DESKVOX_USE_QT4 "Use QT4 not QT5 if available" ON)
endif()
include(Qt4-5)

option(DESKVOX_USE_SSE "Use SSE intrinsic code if possible" ON)

# All options referred to from DeskvoxHandleOptions have to be specified BEFORE this include,
# otherwise options will not be correctly set on first cmake run
include(DeskvoxConfig)
include(DeskvoxHandleOptions)


#---------------------------------------------------------------------------------------------------
# Subdirectories
#


include(DeskvoxAddTarget)
include(DeskvoxUsePackage)


add_subdirectory(virvo)
if(DESKVOX_BUILD_VOX_DESK)
  add_subdirectory(vox-desk-qt)
endif()

if(DESKVOX_BUILD_VOX_DESK_FOX)
  add_subdirectory(vox-desk)
endif()


#---------------------------------------------------------------------------------------------------
# Additional install targets
#


# Copy headers
install(DIRECTORY ${VIRVO_INCLUDE_DIR}/virvo/
  DESTINATION include/virvo
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.hxx"
    PATTERN "*.inl"
    PATTERN "*.inc"
)

# Copy shaders
install(DIRECTORY ${VIRVO_SHADER_DIR}/
  DESTINATION shader
  FILES_MATCHING
    PATTERN "*.fsh"
    PATTERN "*.gsh"
    PATTERN "*.vsh"
    PATTERN "*.glsl"
    PATTERN "*.vert"
    PATTERN "*.frag"
    PATTERN "*.geom"
    PATTERN "*.tess-ctrl"
    PATTERN "*.tess-eval"
)


#---------------------------------------------------------------------------------------------------
# Workaround for MSVS10 to avoid the Dialog Hell
# FIXME: This could be removed with future version of CMake.
#


#if(MSVC_VERSION EQUAL 1600)
#  set(DESKVOX_SLN_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/deskvox.sln")
#  if(EXISTS "${DESKVOX_SLN_FILENAME}")
#    file(APPEND "${DESKVOX_SLN_FILENAME}" "\n# This should be regenerated!\n")
#  endif()
#endif()

IF(${CMAKE_VERSION} VERSION_GREATER "2.8.2" AND NOT COVISE_BUILD)
  INCLUDE(FeatureSummary)
  FEATURE_SUMMARY(WHAT ALL)
ENDIF()
