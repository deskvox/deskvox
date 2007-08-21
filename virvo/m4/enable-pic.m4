AC_DEFUN([ENABLE_PIC],[ 

AC_ARG_ENABLE(pic,
      AC_HELP_STRING([--enable-pic],[build PIC code usable in shared objects]),
      [enable_pic=$enableval],
      [enable_pic=no]
)

if test "$enable_pic" = "yes"; then
      if test -z "`which libtool`" -o "`uname`"="Darwin"; then
          PIC_FLAGS="-fPIC -DPIC"
      else
          PIC_FLAGS=`libtool --config | grep pic_flag|cut -d\" -f2`
      fi

      #echo PIC_FLAGS="$PIC_FLAGS"
      AC_SUBST(PIC_FLAGS)
fi

])
