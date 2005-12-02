#! /bin/sh
autoheader
touch stamp-h.in
aclocal
automake --add-missing
autoconf
./configure $*
