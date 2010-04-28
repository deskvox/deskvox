AC_DEFUN([CHECK_GLEW],[ 

AC_ARG_WITH(glew,
	AC_HELP_STRING([--with-glew],[location of GLEW libraries and headers]),
	[glew_dir=$withval],
	[glew_dir=/usr]
)

glew_libdir="$glew_dir/lib$LIBSUFFIX"
glew_incdir=$glew_dir/include

AC_ARG_WITH(glew-libs,
	AC_HELP_STRING([--with-glew-libs],[location of GLEW libraries]),
	[glew_libdir=$withval],
	[glew_libdir="$glew_dir/lib$LIBSUFFIX"]
)

AC_ARG_WITH(glew-include,
	AC_HELP_STRING([--with-glew-include],[location of GLEW headers]),
	[glew_incdir=$withval],
	[glew_incdir=$glew_dir/include]
)

ac_cppflags_save=$CPPFLAGS
ac_ldflags_save=$LDFLAGS
CPPFLAGS="$CPPFLAGS -I$glew_incdir $GL_INCLUDES"
LDFLAGS="$LDFLAGS -L$glew_libdir $GL_LIBS"
AC_CHECK_HEADERS([GL/glew.h], [have_glew_h=yes])
AC_CHECK_LIB(GLEW, glewIsSupported, [LDFLAGS="$LDFLAGS -lGLEW"; have_libglew=yes])

if test "$have_glew_h" = "yes" -a "$have_libglew" = "yes" ; then
    AC_DEFINE(HAVE_GLEW, 1, [GLEW OpenGL extension wrangler])
    GLEW_INCLUDES="-I$glew_incdir"
    GLEW_LIBS="-L$glew_libdir -lGLEW"
    AC_SUBST(GLEW_INCLUDES)
    AC_SUBST(GLEW_LIBS)
else
    AC_MSG_ERROR([GLEW not found (required)])
fi

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save
])
