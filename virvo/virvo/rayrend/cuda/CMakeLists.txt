if(DESKVOX_USE_CUDA)

find_package(CUDA)
find_package(GLEW REQUIRED)
find_package(Pthreads REQUIRED)

deskvox_use_package(GLEW)
deskvox_use_package(Pthreads)
deskvox_use_package(CUDA INCDIRS ${CUDA_TOOLKIT_INCLUDE})

deskvox_link_libraries(virvo)

set(RAYREND_HEADERS
  ../../vvrayrend.h
  ../../vvrayrend-common.h
)

set(RAYREND_SOURCES
  ../../vvrayrend.cpp
)

add_definitions(-DHAVE_CONFIG_H)


deskvox_cuda_compiles(RAYREND_CUDA_SOURCES
  ../../vvrayrend.cu
)

deskvox_add_library(rayrendcuda
  ${RAYREND_HEADERS}
  ${RAYREND_SOURCES}
  ${RAYREND_CUDA_SOURCES}
)


endif(DESKVOX_USE_CUDA)

