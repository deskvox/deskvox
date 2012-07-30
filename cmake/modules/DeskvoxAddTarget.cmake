include(AddFileDependencies)


#---------------------------------------------------------------------------------------------------
# deskvox_link_libraries(libraries...)
#


macro(deskvox_link_libraries)
  set(__DESKVOX_LINK_LIBRARIES ${__DESKVOX_LINK_LIBRARIES} ${ARGN})
endmacro()


#---------------------------------------------------------------------------------------------------
# deskvox_cuda_compiles(processed_sources, sources...)
#


#function(deskvox_cuda_compiles processed_sources)
#  if(NOT DESKVOX_CUDA_FOUND)
#    return()
#  endif()
#
#  foreach(f ${ARGN})
#    if(BUILD_SHARED_LIBS)
#      cuda_compile(cuda_compile_obj ${f} SHARED)
#    else()
#      cuda_compile(cuda_compile_obj ${f})
#    endif()
#
#    list(APPEND processed_sources ${f})
#    list(APPEND processed_sources ${cuda_compile_obj})
#  endforeach()
#endfunction()


#---------------------------------------------------------------------------------------------------
# __deskvox_process_sources(sources...)
#


function(__deskvox_process_sources)
  foreach(f ${ARGN})
    get_filename_component(path ${f} PATH)

    if(NOT path STREQUAL "")
      string(REPLACE "/" "\\" path "${path}")
      set(group "${path}")
    else()
      set(group "")
    endif()

    source_group("${group}" FILES ${f})

    get_filename_component(ext ${f} EXT)

    if(ext MATCHES "\\.(cu)")
      if(NOT DESKVOX_CUDA_FOUND)
        message(STATUS "Note: '" ${f} "' ignored: CUDA support not available or disabled")
      else()
        if(BUILD_SHARED_LIBS)
          cuda_compile(cuda_compile_obj ${f} SHARED)
        else()
          cuda_compile(cuda_compile_obj ${f})
        endif()

        list(APPEND processed_sources ${f})
        list(APPEND processed_sources ${cuda_compile_obj})
      endif()
    else()
      list(APPEND processed_sources ${f})
    endif()
  endforeach()

  set(__DESKVOX_PROCESSED_SOURCES ${__DESKVOX_PROCESSED_SOURCES} ${processed_sources} PARENT_SCOPE)
endfunction()


#---------------------------------------------------------------------------------------------------
# __deskvox_set_target_postfixes(target)
#


function(__deskvox_set_target_postfixes target)
  #if(BUILD_SHARED_LIBS)
  #  set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "-gd")
  #  set_target_properties(${target} PROPERTIES RELEASE_POSTFIX "")
  #  set_target_properties(${target} PROPERTIES MINSIZEREL_POSTFIX "-m")
  #  set_target_properties(${target} PROPERTIES RELWITHDEBINFO_POSTFIX "-d")
  #else()
  #  set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "$-sgd")
  #  set_target_properties(${target} PROPERTIES RELEASE_POSTFIX "-s")
  #  set_target_properties(${target} PROPERTIES MINSIZEREL_POSTFIX "-sm")
  #  set_target_properties(${target} PROPERTIES RELWITHDEBINFO_POSTFIX "-sd")
  #endif()
endfunction()


#---------------------------------------------------------------------------------------------------
# deskvox_add_library(name, sources...)
#


function(deskvox_add_library name)
  message(STATUS "Adding library " ${name} "...")

  __deskvox_process_sources(${ARGN})

  add_library(${name} ${__DESKVOX_PROCESSED_SOURCES})

  set_target_properties(${name} PROPERTIES FOLDER "Libraries")

  __deskvox_set_target_postfixes(${name})

  target_link_libraries(${name} ${__DESKVOX_LINK_LIBRARIES})

  install(TARGETS ${name}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endfunction()


#---------------------------------------------------------------------------------------------------
# __deskvox_add_executable(folder, name, sources...)
#


function(__deskvox_add_executable folder name)
  message(STATUS "Adding executable: " ${name} " (" ${folder} ")...")

  __deskvox_process_sources(${ARGN})

  add_executable(${name} ${__DESKVOX_PROCESSED_SOURCES})

  set_target_properties(${name} PROPERTIES FOLDER ${folder})

  #__deskvox_set_target_postfixes(${name})

  target_link_libraries(${name} ${__DESKVOX_LINK_LIBRARIES})

  install(TARGETS ${name} RUNTIME DESTINATION bin)
endfunction()


#---------------------------------------------------------------------------------------------------
# deskvox_add_tool(name, sources...)
#


function(deskvox_add_tool name)
  __deskvox_add_executable("Tools" ${name} ${ARGN})
endfunction()


#---------------------------------------------------------------------------------------------------
# deskvox_add_test(name, sources...)
#


function(deskvox_add_test name)
  __deskvox_add_executable("Tests" ${name} ${ARGN})
endfunction()
