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

INCLUDES = $(FOX_INCLUDES) -I../../virvo/src  $(CG_INCLUDES) $(STINGRAY_INCLUDES) 
LIBS = $(FOX_LIBS) -L../../virvo/src -lvirvo  $(CG_LIBS) $(STINGRAY_LIBS)
