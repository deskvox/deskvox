find_package(Boost COMPONENTS serialization REQUIRED)
deskvox_use_package(Boost)

deskvox_add_tool(vconv
  vvconv.cpp
  vvconv.h
)
