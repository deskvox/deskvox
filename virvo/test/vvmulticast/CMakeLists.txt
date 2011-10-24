SET(MULTICASTTEST_SRC vvmulticasttest.cpp)

# vvMulticasttest
ADD_EXECUTABLE(vvmulticasttest ${MULTICASTTEST_SRC})
TARGET_LINK_LIBRARIES(vvmulticasttest virvo)
