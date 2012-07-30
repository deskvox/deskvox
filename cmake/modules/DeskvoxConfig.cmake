include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckCXXSourceCompiles)


if(CMAKE_COMPILER_IS_GNUCXX)
  set(DESKVOX_COMPILER_IS_GCC_COMPATIBLE ON)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(DESKVOX_COMPILER_IS_GCC_COMPATIBLE ON)
endif()

#if(DESKVOX_COMPILER_IS_GCC_COMPATIBLE)
#    add_definitions(-std=c++0x)
#endif()


#---------------------------------------------------------------------------------------------------
# include checks
#

#check_include_file(dlfcn.h HAVE_DLFCN_H)
#check_include_file(execinfo.h HAVE_EXECINFO_H)
#check_include_file(stdint.h HAVE_STDINT_H)
#check_include_file(pthread.h HAVE_PTHREAD_H)


#---------------------------------------------------------------------------------------------------
# library checks
#


#check_library_exists(pthread pthread_create "" HAVE_LIBPTHREAD)


#---------------------------------------------------------------------------------------------------
# function checks
#


#---------------------------------------------------------------------------------------------------
# type checks
#


#function(deskvox_check_type_exists type result)
#  check_cxx_source_compiles(
#    "#include <stdint.h>
#    ${type} var;
#    int main() { return 0; }" ${result})
#endfunction()

#deskvox_check_type_exists(int64_t HAVE_INT64_T)
#deskvox_check_type_exists(uint64_t HAVE_UINT64_T)
