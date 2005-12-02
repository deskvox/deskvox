#! /bin/sh
aclocal
autoheader
automake --add-missing --copy --foreign
autoconf
if [ -f config.status ]; then
   ./config.status --recheck && ./config.status
else
   ./configure $*
fi
