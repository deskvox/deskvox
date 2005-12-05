AC_PREREQ(2.13)
AC_INIT(src/vvvirvo.h)
AC_CONFIG_SRCDIR(configure.ac)
AC_CONFIG_HEADER(src/config.h)

LIBVIRVO_MAJOR_VERSION=2
LIBVIRVO_MINOR_VERSION=0
LIBVIRVO_MICRO_VERSION=2
LIBVIRVO_INTERFACE_AGE=0
LIBVIRVO_EXTRA_VERSION=b
LIBVIRVO_VERSION=$LIBVIRVO_MAJOR_VERSION.$LIBVIRVO_MINOR_VERSION.$LIBVIRVO_MICRO_VERSION

LIBVIRVO_CURRENT=`expr 100 '*' $LIBVIRVO_MINOR_VERSION + $LIBVIRVO_MICRO_VERSION - $LIBVIRVO_INTERFACE_AGE`
LIBVIRVO_BINARY_AGE=`expr 100 '*' $LIBVIRVO_MINOR_VERSION + $LIBVIRVO_MICRO_VERSION`
LIBVIRVO_REVISION=$LIBVIRVO_INTERFACE_AGE
LIBVIRVO_AGE=`expr $LIBVIRVO_BINARY_AGE - $LIBVIRVO_INTERFACE_AGE`
LIBVIRVO_VERSION=$LIBVIRVO_MAJOR_VERSION.$LIBVIRVO_MINOR_VERSION.$LIBVIRVO_MICRO_VERSION$LIBVIRVO_EXTRAVERSION

AC_SUBST(LIBVIRVO_VERSION)

AM_INIT_AUTOMAKE(libvirvo, $LIBVIRVO_VERSION)

AC_SUBST(ACLOCAL_AMFLAGS, "$ACLOCAL_FLAGS")

AM_MAINTAINER_MODE

AC_PROG_CC
AC_PROG_CXX
AC_PROG_LD
AC_PROG_INSTALL
AC_PROG_LIBTOOL

AC_ARG_ENABLE(cg,
	AC_HELP_STRING([--enable-cg],[build virvo with Cg shaders]),
	[enable_cg=$enableval],
	[enable_cg=yes]
)

AC_ARG_WITH(cg,
	AC_HELP_STRING([--with-cg],[location of Cg libraries and headers]),
	[cg_dir=$withval],
	[cg_dir=/usr]
)

cg_libdir=$cg_dir/lib
cg_incdir=$cg_dir/include

AC_ARG_WITH(cg-libs,
	AC_HELP_STRING([--with-cg-libs],[location of Cg libraries]),
	[cg_libdir=$withval],
	[cg_libdir=$cg_dir/lib]
)

AC_ARG_WITH(cg-include,
	AC_HELP_STRING([--with-cg-include],[location of Cg headers]),
	[cg_incdir=$withval],
	[cg_incdir=$cg_dir/include]
)

if test "$enable_cg" != "no"; then
    ac_cppflags_save=$CPPFLAGS
    CPPFLAGS="$CPPFLAGS -I$cg_incdir"
    LDFLAGS="$LDFLAGS -L$cg_libdir"
    AC_CHECK_HEADERS([Cg/cg.h], [have_cg_h=yes])
    AC_CHECK_LIB(m, log, LDFLAGS="$LDFLAGS -lm")
    AC_CHECK_LIB(pthread, pthread_once, LDFLAGS="$LDFLAGS -lpthread")
    AC_CHECK_LIB(GL, glGetIntegerv, LDFLAGS="$LDFLAGS -lGL")
    AC_CHECK_LIB(GLU, gluSphere, LDFLAGS="$LDFLAGS -lGLU")
    AC_CHECK_LIB(Cg, cgSetParameter1d, LDFLAGS="$LDFLAGS -lCg")
    AC_CHECK_LIB(CgGL, cgGLSetParameter1d, have_libcggl=yes)
    CPPFLAGS=$ac_cppflags_save
    LDFLAGS=$ac_ldflags_save

    if test "$have_cg_h" = "yes" -a "$have_libcggl" = "yes" ; then
        AC_DEFINE(HAVE_CG, 1, [Cg shaders enabled])
        CG_INCLUDES="-I$cg_incdir"
        CG_LIBS="-L$cg_libdir -lCgGL -lCg"
        AC_SUBST(CG_INCLUDES)
        AC_SUBST(CG_LIBS)
    fi
fi

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


AC_SUBST(CPPFLAGS)
AC_SUBST(CFLAGS)
AC_SUBST(CXXFLAGS)
AC_SUBST(LDFLAGS)

AC_OUTPUT(Makefile src/Makefile)


