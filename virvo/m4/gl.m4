AC_DEFUN([CHECK_GL],[ 

AC_ARG_WITH(gl,
	AC_HELP_STRING([--with-gl],[location of GL libraries and headers]),
	[gl_dir=$withval],
	[gl_dir=/usr]
)

gl_libdir=$gl_dir/lib
gl_incdir=$gl_dir/include

AC_ARG_WITH(gl-libs,
	AC_HELP_STRING([--with-gl-libs],[location of OpenGL libraries]),
	[gl_libdir=$withval],
	[gl_libdir=$gl_dir/lib]
)

AC_ARG_WITH(gl-include,
	AC_HELP_STRING([--with-gl-include],[location of OpenGL headers]),
	[gl_incdir=$withval],
	[gl_incdir=$gl_dir/include]
)

ac_cppflags_save=$CPPFLAGS
ac_ldflags_save=$LDFLAGS
CPPFLAGS="$CPPFLAGS -I$gl_incdir"
LDFLAGS="$LDFLAGS -L$gl_libdir"
AC_CHECK_HEADERS([GL/gl.h], [have_gl_gl_h=yes])
AC_CHECK_LIB(GL, glGetString, [LDFLAGS="$LDFLAGS -lGL"; have_libgl=yes])

if test "$have_gl_gl_h" = "yes" -a "$have_libgl" = "yes" ; then
    AC_DEFINE(HAVE_GL, 1, [OpenGL library])
    GL_INCLUDES="-I$gl_incdir"
    GL_LIBS="-L$gl_libdir -lGL"
    AC_SUBST(GL_INCLUDES)
    AC_SUBST(GL_LIBS)
else
    CPPFLAGS=$ac_cppflags_save
    LDFLAGS=$ac_ldflags_save

    AC_CHECK_HEADERS([OpenGL/gl.h], [have_opengl_gl_h=yes])
    LDFLAGS="$LDFLAGS -framework OpenGL"
    AC_CHECK_FUNC(glGetString, [have_gl_framework=yes])
    if test "$have_opengl_gl_h" = "yes" -a "$have_gl_framework" = "yes" ; then
        AC_DEFINE(HAVE_GL_FRAMEWORK, 1, [OpenGL framework])
        if test -n "$gl_dir"; then
            GL_INCLUDES="-F$gl_dir"
        fi
        GL_LIBS="-framework OpenGL"
    else
        AC_MSG_ERROR([OpenGL not found (required)])
    fi
fi

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save
])
