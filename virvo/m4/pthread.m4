AC_DEFUN([CHECK_PTHREAD],[ 

AC_ARG_WITH(pthread,
	AC_HELP_STRING([--with-pthread],[location of POSIX thread libraries and headers]),
	[pthread_dir=$withval],
	[pthread_dir=/usr]
)

pthread_libdir="$pthread_dir/lib$LIBSUFFIX"
pthread_incdir=$pthread_dir/include

AC_ARG_WITH(pthread-libs,
	AC_HELP_STRING([--with-pthread-libs],[location of POSIX thread libraries]),
	[pthread_libdir=$withval],
	[pthread_libdir="$pthread_dir/lib$LIBSUFFIX"]
)

AC_ARG_WITH(pthread-include,
	AC_HELP_STRING([--with-pthread-include],[location of POSIX thread library headers]),
	[pthread_incdir=$withval],
	[pthread_incdir=$pthread_dir/include]
)

ac_cppflags_save=$CPPFLAGS
ac_ldflags_save=$LDFLAGS
CPPFLAGS="$CPPFLAGS -I$pthread_incdir"
LDFLAGS="$LDFLAGS -L$pthread_libdir"
AC_CHECK_HEADERS([pthread.h], [have_pthread_h=yes])
AC_CHECK_LIB(pthread, pthread_create, [LDFLAGS="$LDFLAGS -lpthread"; have_libpthread=yes])

if test "$have_pthread_h" = "yes" -a "$have_libpthread" = "yes" ; then
    AC_DEFINE(HAVE_PTHREAD, 1, [POSIX thread library])
    PTHREAD_INCLUDES="-I$pthread_incdir"
    PTHREAD_LIBS="-L$pthread_libdir -lpthread"
    AC_SUBST(PTHREAD_INCLUDES)
    AC_SUBST(PTHREAD_LIBS)
fi

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save
])
