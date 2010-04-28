AC_DEFUN([CHECK_LIBDL],[ 

AC_ARG_WITH(libdl,
	AC_HELP_STRING([--with-libdl],[location of POSIX thread libraries and headers]),
	[libdl_dir=$withval],
	[libdl_dir=/usr]
)

libdl_libdir="$libdl_dir/lib$LIBSUFFIX"
libdl_incdir=$libdl_dir/include

AC_ARG_WITH(libdl-libs,
	AC_HELP_STRING([--with-libdl-libs],[location of POSIX thread libraries]),
	[libdl_libdir=$withval],
	[libdl_libdir="$libdl_dir/lib$LIBSUFFIX"]
)

ac_cppflags_save=$CPPFLAGS
ac_ldflags_save=$LDFLAGS
CPPFLAGS="$CPPFLAGS -I$libdl_incdir"
LDFLAGS="$LDFLAGS -L$libdl_libdir"
AC_CHECK_LIB(dl, dlopen, [LDFLAGS="$LDFLAGS -ldl"; have_libdl=yes])

if test "$have_libdl" = "yes" ; then
    AC_DEFINE(HAVE_LIBDL, 1, [dynamic linking library])
    LIBDL_LIBS="-L$libdl_libdir -ldl"
    AC_SUBST(LIBDL_LIBS)
fi

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save
])
