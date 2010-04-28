AC_DEFUN([CHECK_LIBSUFFIX],[ 

AC_ARG_WITH(libsuffix,
	AC_HELP_STRING([--with-libsuffix],[suffix to append to /lib subdirectories]),
	[libsuffix=$withval],
	[libsuffix=]
)

LIBSUFFIX="$libsuffix"
])
