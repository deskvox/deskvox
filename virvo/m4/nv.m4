AC_DEFUN([CHECK_NV],[ 

AC_ARG_ENABLE(nv,
        AC_HELP_STRING([--enable-nv],[build virvo with proprietary code from nvidia company]),
        [enable_nv=$enableval],
        [enable_nv=yes]
)

if test "$enable_nv" != "no"; then
	AC_DEFINE(NV_PROPRIETARY_CODE, 1, [Use proprietary code from nvidia company])
fi
])
