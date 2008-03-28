AC_DEFUN([CHECK_FOX],[ 

AC_ARG_WITH(fox,
	AC_HELP_STRING([--with-fox],[location of FOX toolkit libraries and headers]),
	[fox_dir=$withval]
)

if test -n "$fox_dir"; then
    fox_config=$fox_dir/bin/fox-config
fi

AC_ARG_WITH(fox-config,
	AC_HELP_STRING([--with-fox-config],[location of FOX toolkit config script]),
	[fox_config=$withval]
)

fox_config=${fox_config:-fox-config}

if test -e "${fox_config}"; then
   AC_MSG_RESULT(using ${fox_config})
else
   AC_PATH_PROG(fox_config, ${fox_config})
   if test ! -e "${fox_config}"; then
       AC_MSG_ERROR(fox-config not found)
   fi
fi

ac_cppflags_save="$CPPFLAGS"
ac_ldflags_save="$LDFLAGS"

AC_MSG_CHECKING(for FOX CFLAGS)
FOX_INCLUDES="`${fox_config} --cflags`"
AC_MSG_RESULT($FOX_INCLUDES)

AC_MSG_CHECKING(for FOX LDFLAGS)
FOX_LIBS="`${fox_config} --libs`"
AC_MSG_RESULT($FOX_LIBS)

CPPFLAGS="${CPPFLAGS} $FOX_INCLUDES"
LDFLAGS="$LDFLAGS $FOX_LIBS"

AC_SUBST(FOX_INCLUDES)
AC_SUBST(FOX_LIBS)

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save

])
