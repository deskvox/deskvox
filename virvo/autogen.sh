#! /bin/sh
aclocal -I m4 || exit 1
libtoolize || glibtoolize || exit 1
autoheader || exit 1
automake --add-missing --copy --foreign || exit 1
autoconf || exit 1

if [ -z "$*" -a -f config.status ]; then
   ./config.status --recheck && ./config.status
else
   ./configure $*
fi
