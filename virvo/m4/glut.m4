AC_DEFUN([CHECK_GLUT],[ 

AC_ARG_WITH(glut,
	AC_HELP_STRING([--with-glut],[location of GLUT libraries and headers]),
	[glut_dir=$withval],
	[glut_dir=/usr]
)

glut_libdir=$glut_dir/lib
glut_incdir=$glut_dir/include

AC_ARG_WITH(glut-libs,
	AC_HELP_STRING([--with-glut-libs],[location of GLUT libraries]),
	[glut_libdir=$withval],
	[glut_libdir=$glut_dir/lib]
)

AC_ARG_WITH(glut-include,
	AC_HELP_STRING([--with-glut-include],[location of GLUT headers]),
	[glut_incdir=$withval],
	[glut_incdir=$glut_dir/include]
)

ac_cppflags_save=$CPPFLAGS
ac_ldflags_save=$LDFLAGS
CPPFLAGS="$CPPFLAGS -I$glut_incdir"
LDFLAGS="$LDFLAGS -L$glut_libdir"
AC_CHECK_HEADERS([GL/glut.h], [have_glut_h=yes])
AC_CHECK_LIB(glut, glutInit, [LDFLAGS="$LDFLAGS -lglut"; have_libglut=yes])

if test "$have_glut_h" = "yes" -a "$have_libglut" = "yes" ; then
    AC_DEFINE(HAVE_GLUT, 1, [GLUT OpenGL extension wrangler])
    GLUT_INCLUDES="-I$glut_incdir"
    GLUT_LIBS="-L$glut_libdir -lglut"
    AC_SUBST(GLUT_INCLUDES)
    AC_SUBST(GLUT_LIBS)
fi

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save
])
