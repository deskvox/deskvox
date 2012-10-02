INCLUDE_DIRECTORIES(../vcommon)

find_package(Pthreads REQUIRED)

deskvox_use_package(Pthreads)

deskvox_add_tool(vserver
  vvresourcemanager.cpp
  vvresourcemanager.h
  vvserver.h
  vvserver.cpp
  vvsimpleserver.h
  vvsimpleserver.cpp
)
