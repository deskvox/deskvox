AC_DEFUN([CHECK_STINGRAY],[ 

AC_ARG_ENABLE(stingray,
      AC_HELP_STRING([--enable-stingray],[build with Stingray raycaster]),
      [enable_stingray=$enableval],
      [enable_stingray=no]
)

AC_ARG_WITH(stingray,
	AC_HELP_STRING([--with-stingray],[location of stingray library and headers]),
	[stingray_dir=$withval],
	[stingray_dir=/usr/local]
)

if test "$enable_stingray" = "yes"; then
        AC_DEFINE(HAVE_STINGRAY, 1, [Stingray raycaster enabled])
        STINGRAY_INCLUDES="-I$stingray_dir/include -I$stingray_dir/include/StingRayCave -I$stingray_dir/include/DataObject"
        STINGRAY_LIBS="-L$stingray_libdir -lstingray"
        AC_SUBST(STINGRAY_INCLUDES)
        AC_SUBST(STINGRAY_LIBS)
fi

])
