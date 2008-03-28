#! /bin/sh
aclocal -I virvo/m4 -I vox-desk/m4
autoheader
automake --add-missing --copy --foreign
autoconf
if [ -z "$*" -a -f config.status ]; then
   ./config.status --recheck && ./config.status
else
   MACHINE=`uname -m`
   case $MACHINE in
      x86_64*) ARCH=amd64 ;;
      i386*|i486*|i586*|i686*) ARCH=gcc3 ;;
   esac
   
   if [ -z "$*" -a ! -z "$ARCH" ]; then
       ./configure --enable-cg --with-cg=`pwd` --with-cg-libs=`pwd`/lib/$ARCH
   else
       ./configure $*
   fi
fi
