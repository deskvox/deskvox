AUTOMAKE_OPTIONS = foreign

EXTRA_DIST =

DISTCLEANFILES =

lib_LIBRARIES = libvirvo.a

libvirvo_a_headers = \
	glext-orig.h \
	vvarray.h \
        vvcolor.h \
	vvdebugmsg.h \
	vvdicom.h \
	vvdynlib.h \
	vvexport.h \
	vvfileio.h \
	vvglext.h \
	vvgltools.h \
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
	vvglsl.h \
	vvmultirend/vvtexmultirend.h \
	vvmultirend/vvtexmultirendmngr.h

libvirvo_a_SOURCES = \
	$(libvirvo_a_headers) \
        vvcolor.cpp \
	vvdebugmsg.cpp \
	vvdicom.cpp \
	vvdynlib.cpp \
	vvfileio.cpp \
	vvgltools.cpp \
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
	vvvoldesc.cpp \
	vvglsl.cpp \
	vvmultirend/vvtexmultirend.cpp \
	vvmultirend/vvtexmultirendmngr.cpp

libvirvoincludedir = $(includedir)/virvo
libvirvoinclude_HEADERS = $(libvirvo_a_headers) config.h

INCLUDES=$(LIBVIRVO_CFLAGS) $(CG_INCLUDES) $(STINGRAY_INCLUDES) $(PIC_FLAGS)
LIBS=$(LIBVIRVO_LIBS) $(CG_LIBS) $(STINGRAY_LIBS)
