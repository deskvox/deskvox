find_package(Qt REQUIRED)
include_directories(${VOX_DESK_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/virvo)
include_directories(${QT_INCLUDE_DIR} ${QT_QTCORE_INCLUDE_DIR} ${QT_QTGUI_INCLUDE_DIR} ${QT_QTOPENGL_INCLUDE_DIR})
add_definitions(-DQT_PLUGIN)

add_subdirectory(servermon)
