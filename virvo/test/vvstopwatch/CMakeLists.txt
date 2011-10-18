SET(STOPWATCHTEST_SRC vvstopwatchtest.cpp)

ADD_EXECUTABLE(stopwatchtest ${STOPWATCHTEST_SRC})
TARGET_LINK_LIBRARIES(stopwatchtest virvo)

