AC_DEFUN([CHECK_XLIBS],[ 

AC_ARG_WITH(xlibs,
	AC_HELP_STRING([--with-xlibs],[location of X11 libraries and headers]),
	[xlibs_dir=$withval],
	[xlibs_dir=/usr]
)

xlibs_libdir="$xlibs_dir/lib$LIBSUFFIX"
xlibs_incdir=$xlibs_dir/include

AC_ARG_WITH(xlibs-libs,
	AC_HELP_STRING([--with-xlibs-libs],[location of X11 libraries]),
	[xlibs_libdir=$withval],
	[xlibs_libdir="$xlibs_dir/lib$LIBSUFFIX"]
)

AC_ARG_WITH(xlibs-include,
	AC_HELP_STRING([--with-xlibs-include],[location of X11 library headers]),
	[xlibs_incdir=$withval],
	[xlibs_incdir=$xlibs_dir/include]
)

ac_cppflags_save=$CPPFLAGS
ac_ldflags_save=$LDFLAGS
CPPFLAGS="$CPPFLAGS -I$xlibs_incdir"
LDFLAGS="$LDFLAGS -L$xlibs_libdir"
AC_CHECK_HEADERS([X11/X.h], [have_xlibs_headers=yes])
AC_CHECK_LIB(X11, XFlush, [LDFLAGS="$LDFLAGS -lX11"; have_xlibs=yes])

if test "$have_xlibs_headers" = "yes" -a "$have_xlibs" = "yes" ; then
    AC_DEFINE(HAVE_XLIBS, 1, [X11 libraries])
    XLIBS_INCLUDES="-I$xlibs_incdir"
    XLIBS_LIBS="-L$xlibs_libdir -lX11"
    AC_SUBST(XLIBS_INCLUDES)
    AC_SUBST(XLIBS_LIBS)
fi

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save
])
