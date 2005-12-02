#! /bin/sh
aclocal
automake
autoconf
./configure $*
