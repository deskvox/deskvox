AC_DEFUN([CHECK_CG],[ 

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

AC_ARG_WITH(cg-framework,
	AC_HELP_STRING([--with-cg-framework],[location of Cg framework (Mac OS X)]),
	[cg_framework=$withval],
	[cg_framework=yes]
)

if test "$enable_cg" != "no"; then
    ac_cppflags_save=$CPPFLAGS
    ac_ldflags_save=$LDFLAGS
    if test "$cg_framework" != "no"; then
        if test "$cg_framework" != "" -a "$cg_framework" != "yes"; then
            cg_framework_path="-F$cg_framework"
        fi
        CPPFLAGS="$CPPFLAGS $cg_framework_path -framework Cg"
        LDFLAGS="$LDFLAGS $cg_framework_path -framework Cg"
        AC_CHECK_HEADERS([Cg/cg.h], [have_cg_h=yes])

        if test "$have_cg_h" = "yes"; then
            AC_DEFINE(HAVE_CG, 1, [Cg shaders enabled])
            CG_INCLUDES="$cg_framework_path"
            # only for shared libs
            CG_LIBS="$cg_framework_path -framework Cg"
            AC_SUBST(CG_INCLUDES)
            AC_SUBST(CG_LIBS)
        else
            CPPFLAGS=$ac_cppflags_save
            LDFLAGS=$ac_ldflags_save
        fi
    fi

    if test "$have_cg_h" != "yes"; then
        CPPFLAGS="$CPPFLAGS -I$cg_incdir"
        LDFLAGS="$LDFLAGS -L$cg_libdir"
        AC_CHECK_HEADERS([Cg/cg.h], [have_cg_h=yes])
        AC_CHECK_LIB(m, log, [LDFLAGS="$LDFLAGS -lm"])
        AC_CHECK_LIB(pthread, pthread_once, [LDFLAGS="$LDFLAGS -lpthread"])
        AC_CHECK_LIB(GL, glGetIntegerv, [LDFLAGS="$LDFLAGS -lGL"])
        AC_CHECK_LIB(GLU, gluSphere, [LDFLAGS="$LDFLAGS -lGLU"])
        AC_CHECK_LIB(Cg, cgSetParameter1d, [LDFLAGS="$LDFLAGS -lCg"])
        AC_CHECK_LIB(CgGL, cgGLSetParameter1d, have_libcggl=yes)

        if test "$have_cg_h" = "yes" -a "$have_libcggl" = "yes" ; then
            AC_DEFINE(HAVE_CG, 1, [Cg shaders enabled])
            CG_INCLUDES="-I$cg_incdir"
            CG_LIBS="-rpath $cg_libdir -L$cg_libdir -lCgGL -lCg"
            AC_SUBST(CG_INCLUDES)
            AC_SUBST(CG_LIBS)
        fi
    fi

    CPPFLAGS=$ac_cppflags_save
    LDFLAGS=$ac_ldflags_save
fi

])
