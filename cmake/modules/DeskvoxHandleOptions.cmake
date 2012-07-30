# This CMake module is responsible for interpreting the user defined DESKVOX_* options and
# executing the appropriate CMake commands to realize the users' selections.


if(DESKVOX_USE_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()


if(NOT BUILD_SHARED_LIBS)
  add_definitions(-DNODLL)
endif()


if(MSVC)

  add_definitions(
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_SECURE_NO_WARNINGS
    -D_CRT_NONSTDC_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_WARNINGS
    -D_SCL_SECURE_NO_DEPRECATE
    -D_SCL_SECURE_NO_WARNINGS

    /wd4251 # Disable: "'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'"
    /wd4275 # Disable: "non-DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'"
    /wd4503 # Disable: "'identifier' : decorated name length exceeded, name was truncated"

    /w14062 # Promote to level 1 warning: "enumerator in switch of enum is not handled"
    /w14146 # Promote to level 1 warning: "unary minus operator applied to unsigned type, result still unsigned"

    #
    # Promote warnings to errors:
    #
    # Using warning-as-errors will make it hard to miss or ignore warnings. Microsoft command line option /WX, makes
    # all warnings errors, but this may be too drastic a step.
    # The following have been recommended by Pete Bartlett to make the MSVC and GCC compilers behave as similarly as possible:
    #

    /we4238 # Don't take address of temporaries
    /we4239 # Don't bind temporaries to non-const references (Stephan's "Evil Extension")
    /we4288 # For-loop scoping (this is the default)
    /we4346 # Require "typename" where the standard requires it
  )

  # Note:
  # pedantic is not supported - enable all warnings instead
  if(DESKVOX_ENABLE_WARNINGS OR DESKVOX_ENABLE_PEDANTIC)
    add_definitions(/W4)
  endif()
  if(DESKVOX_ENABLE_WERROR)
    add_definitions(/WX)
  endif()

elseif(DESKVOX_COMPILER_IS_GCC_COMPATIBLE)

  add_definitions(-Wmissing-braces)
  add_definitions(-Wsign-compare)
  add_definitions(-Wwrite-strings)
  add_definitions(-Woverloaded-virtual)

  if(DESKVOX_ENABLE_WARNINGS)
    add_definitions(-Wall -Wextra)
  endif()
  if(DESKVOX_ENABLE_PEDANTIC)
    add_definitions(-pedantic)
  endif()
  if(DESKVOX_ENABLE_WERROR)
    add_definitions(-Werror)
  endif()

endif()
