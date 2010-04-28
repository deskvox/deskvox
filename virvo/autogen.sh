#! /bin/sh
aclocal -I m4
libtoolize || glibtoolize
autoheader
automake --add-missing --copy --foreign
autoconf

if [ -z "$*" -a -f config.status ]; then
   ./config.status --recheck && ./config.status
else
   ./configure $*
fi
