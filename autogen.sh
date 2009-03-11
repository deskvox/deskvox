#! /bin/sh
aclocal -I virvo/m4 -I vox-desk/m4
libtoolize
autoheader
automake --add-missing --copy --foreign
autoconf

if [ -z "$*" -a -f config.status ]; then
   ./config.status --recheck && ./config.status
else
   ./configure $*
fi
