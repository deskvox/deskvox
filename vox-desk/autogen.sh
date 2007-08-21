#! /bin/sh
autoheader
touch stamp-h.in
aclocal -I ../virvo/m4 -I m4
automake --add-missing 
autoconf
./configure $*
