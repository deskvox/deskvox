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

vconv deskvox: ../../virvo/src/libvirvo.a

INCLUDES = $(FOX_INCLUDES) -I$(abs_top_srcdir)/virvo/src $(CUDA_INCLUDES) $(CG_INCLUDES) $(GLEW_INCLUDES) $(STINGRAY_INCLUDES) 
LIBS = $(FOX_LIBS) -L../../virvo/src -lvirvo $(CUDA_LIBS) $(CG_LIBS) $(GLEW_LIBS) $(STINGRAY_LIBS)
