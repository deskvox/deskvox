find_package(Boost COMPONENTS chrono system thread REQUIRED)

deskvox_use_package(Boost)

add_definitions(-DHAVE_CONFIG_H)

deskvox_add_tool(vserver_asio
  main.cpp
  message_queue.h
  resource_manager.cpp
  resource_manager.h
  server.cpp
  server.h
  server_manager.cpp
  server_manager.h
  simple_server.cpp
  simple_server.h
)
