EXTRA_DIST =

DISTCLEANFILES =

lib_LTLIBRARIES = libvirvo.la
libvirvo_la_SOURCES = \
	glext-orig.h \
	vvarray.h \
	vvdebugmsg.h \
	vvdicom.h \
	vvdynlib.h \
	vvexport.h \
	vvfileio.h \
	vvglext.h \
	vvgltools.h \
	vvideo.h \
	vvimage.h \
	vvopengl.h \
	vvprintgl.h \
	vvrenderer.h \
	vvsllist.h \
	vvsocket.h \
	vvsocketio.h \
	vvsphere.h \
	vvstingray.h \
	vvstopwatch.h \
	vvtexrend.h \
	vvtfwidget.h \
	vvtokenizer.h \
	vvtoolshed.h \
	vvtransfunc.h \
	vvvecmath.h \
	vvvffile.h \
	vvvirvo.h \
	vvvoldesc.h \
	\
	vvdebugmsg.cpp \
	vvdicom.cpp \
	vvdynlib.cpp \
	vvfileio.cpp \
	vvgltools.cpp \
	vvideo.cpp \
	vvimage.cpp \
	vvprintgl.cpp \
	vvrenderer.cpp \
	vvsocket.cpp \
	vvsocketio.cpp \
	vvsphere.cpp \
	vvstingray.cpp \
	vvstopwatch.cpp \
	vvtexrend.cpp \
	vvtfwidget.cpp \
	vvtokenizer.cpp \
	vvtoolshed.cpp \
	vvtransfunc.cpp \
	vvvecmath.cpp \
	vvvffile.cpp \
	vvvoldesc.cpp

libvirvo_la_headers = \
	glext-orig.h \
	vvarray.h \
	vvdebugmsg.h \
	vvdicom.h \
	vvdynlib.h \
	vvexport.h \
	vvfileio.h \
	vvglext.h \
	vvgltools.h \
	vvideo.h \
	vvimage.h \
	vvopengl.h \
	vvprintgl.h \
	vvrenderer.h \
	vvsllist.h \
	vvsocket.h \
	vvsocketio.h \
	vvsphere.h \
	vvstingray.h \
	vvstopwatch.h \
	vvtexrend.h \
	vvtfwidget.h \
	vvtokenizer.h \
	vvtoolshed.h \
	vvtransfunc.h \
	vvvecmath.h \
	vvvffile.h \
	vvvirvo.h \
	vvvoldesc.h

libvirvo_la_LDFLAGS = -release $(LIBVIRVO_VERSION) -no-undefined

libvirvoincludedir = $(includedir)
libvirvoinclude_HEADERS = $(libvirvo_la_headers)

INCLUDES=$(LIBVIRVO_CFLAGS) $(CG_INCLUDES) $(STINGRAY_INCLUDES)
LIBS=$(LIBVIRVO_LIBS) $(CG_LIBS) $(STINGRAY_LIBS)
