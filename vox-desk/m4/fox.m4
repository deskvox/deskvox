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

if test -z "$fox_config"; then
   fox_config=fox-config
fi

ac_cppflags_save="$CPPFLAGS"
ac_ldflags_save="$LDFLAGS"

FOX_INCLUDES="`${fox_config} --cflags`"
FOX_LIBS="`${fox_config} --libs`"

CPPFLAGS="${CPPFLAGS} $FOX_INCLUDES"
LDFLAGS="$LDFLAGS $FOX_LIBS"

AC_SUBST(FOX_INCLUDES)
AC_SUBST(FOX_LIBS)
echo $CPPFLAGS
echo $LDFLAGS

CPPFLAGS=$ac_cppflags_save
LDFLAGS=$ac_ldflags_save

])
