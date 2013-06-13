find_package(Boost)
find_package(GLEW REQUIRED)
find_package(Pthreads REQUIRED)

deskvox_use_package(Boost)
deskvox_use_package(GLEW)
deskvox_use_package(Pthreads)

deskvox_link_libraries(virvo)

set(RAYREND_HEADERS
  ../../vvsoftrayrend.h
)

set(RAYREND_SOURCES
  ../../vvsoftrayrend.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
add_definitions(-DHAVE_CONFIG_H)
add_definitions(-DVV_USE_SSE=1)

deskvox_add_library(rayrendsse
  ${RAYREND_HEADERS}
  ${RAYREND_SOURCES}
)

