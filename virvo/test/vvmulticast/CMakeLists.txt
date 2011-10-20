SET(MULTICASTTEST_SRC vvmulticasttest.cpp)

# vvMulticasttest
ADD_EXECUTABLE(multicasttest ${MULTICASTTEST_SRC})
TARGET_LINK_LIBRARIES(multicasttest virvo)
