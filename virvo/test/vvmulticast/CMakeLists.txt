SET(MULTICASTTEST_SRC vvmulticasttest.cpp)
SET(MULTICASTTEST2_SRC vvmulticasttest2.cpp)

# vvMulticasttest
INCLUDE_DIRECTORIES(${PTHREAD_INCLUDE_DIR})
ADD_EXECUTABLE(vvmulticasttest ${MULTICASTTEST_SRC})
TARGET_LINK_LIBRARIES(vvmulticasttest virvo ${PTHREAD_LIBRARIES})

# vvMulticasttest2 (with vvSocket)
ADD_EXECUTABLE(vvmulticasttest2 ${MULTICASTTEST2_SRC})
TARGET_LINK_LIBRARIES(vvmulticasttest2 virvo)
