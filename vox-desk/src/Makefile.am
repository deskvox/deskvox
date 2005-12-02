AUTOMAKE_OPTIONS = foreign

bin_PROGRAMS = deskvox vconv

deskvox_SOURCES = \
	vvcanvas.cpp \
	vvmovie.cpp \
	vvprefwindow.cpp \
	vvsliceviewer.cpp \
	vvtranswindow.cpp \
	vvcanvas.h \
	vvmovie.h \
	vvprefwindow.h \
	vvsliceviewer.h \
	vvtranswindow.h \
	vvartracking.cpp \
	vvclient.cpp \
	vvdialogs.cpp \
	vvobjview.cpp \
	vvshell.cpp \
	vvartracking.h \
	vvclient.h \
	vvdialogs.h \
	vvobjview.h \
	vvshell.h

vconv_SOURCES = \
	vvconv.cpp \
	vvconv.h

INCLUDES = -DHAVE_CG -I../../virvo/src -I../../virvo/src -I ../../virvo/include -I../include/fox-1_4
LIBS = -L../lib/amd64 -L/usr/X11R6/lib64 -L../../virvo/src -L../../virvo/lib/amd64 -lvirvo -lX11 -lXrandr -lXcursor -lFOX-1.4 -lGLU -lGL -lCgGL -lCg -lpng -lpthread
