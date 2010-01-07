AC_DEFUN([CHECK_GLU],[ 

AC_ARG_WITH(glu,
	AC_HELP_STRING([--with-glu],[location of GLU libraries and headers]),
	[glu_dir=$withval],
	[glu_dir=/usr]
)

glu_libdir=$glu_dir/lib
glu_incdir=$glu_dir/include

AC_ARG_WITH(glu-libs,
	AC_HELP_STRING([--with-glu-libs],[location of GLU libraries]),
	[glu_libdir=$withval],
	[glu_libdir=$glu_dir/lib]
)

AC_ARG_WITH(glu-include,
	AC_HELP_STRING([--with-glu-include],[location of GLU headers]),
	[glu_incdir=$withval],
	[glu_incdir=$glu_dir/include]
)

ac_cppflags_save=$CPPFLAGS
ac_ldflags_save=$LDFLAGS
CPPFLAGS="$CPPFLAGS -I$glu_incdir"
LDFLAGS="$LDFLAGS -L$glu_libdir"
AC_CHECK_HEADERS([GL/glu.h], [have_glu_h=yes])
AC_CHECK_LIB(GLU, gluSphere, [LDFLAGS="$LDFLAGS -lGLU"; have_libglu=yes])

if test "$have_glu_h" = "yes" -a "$have_libglu" = "yes" ; then
    AC_DEFINE(HAVE_GLU, 1, [GLU OpenGL utility library])
    GLU_INCLUDES="-I$glu_incdir"
    GLU_LIBS="-L$glu_libdir -lGLU"
    AC_SUBST(GLU_INCLUDES)
    AC_SUBST(GLU_LIBS)
fi

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save
])
