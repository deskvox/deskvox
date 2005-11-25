#! /bin/sh
aclocal
autoheader
libtoolize --automake
automake --add-missing --copy --foreign
autoconf
./configure $*
