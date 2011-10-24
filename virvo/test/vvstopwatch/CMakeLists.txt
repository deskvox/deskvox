SET(STOPWATCHTEST_SRC vvstopwatchtest.cpp)

ADD_EXECUTABLE(vvstopwatchtest ${STOPWATCHTEST_SRC})
TARGET_LINK_LIBRARIES(vvstopwatchtest virvo)

