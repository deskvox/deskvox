find_package(Pthreads REQUIRED)

deskvox_use_package(Pthreads)

deskvox_add_tool(vserver
  vvresourcemanager.cpp
  vvresourcemanager.h
  vvserver.cpp
  vvserver.h
)
