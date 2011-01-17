AC_DEFUN([CHECK_VOLPACK],[ 

AC_ARG_ENABLE(volpack,
      AC_HELP_STRING([--enable-volpack],[build with VolPack software renderer]),
      [enable_volpack=$enableval],
      [enable_volpack=no]
)

AC_ARG_WITH(volpack,
	AC_HELP_STRING([--with-volpack],[location of volpack library and headers]),
	[volpack_dir=$withval],
	[volpack_dir=/usr]
)

if test "$enable_volpack" = "yes"; then
        AC_DEFINE(HAVE_VOLPACK, 1, [VolPack software renderer enabled])
        AC_DEFINE(USE_VOLPACK, 1, [VolPack software renderer enabled])
        VOLPACK_INCLUDES="-I$volpack_dir/include -I$volpack_dir/include -I$volpack_dir/include"
        VOLPACK_LIBS="-L$volpack_dir/lib -lvolpack"
        AC_SUBST(VOLPACK_INCLUDES)
        AC_SUBST(VOLPACK_LIBS)
fi

])
