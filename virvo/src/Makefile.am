AUTOMAKE_OPTIONS = foreign

EXTRA_DIST =

DISTCLEANFILES =

lib_LIBRARIES = libvirvo.a

libvirvo_a_headers = \
        glext-orig.h \
        vvarray.h \
        vvbrick.h \
        vvbsptree.h \
        vvcg.h \
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
        vvshadermanager.h \
        vvshaderfactory.h \
        vvrenderslave.h \
        vvrendertarget.h \
        vvrendercontext.h \
        vvoffscreenbuffer.h \
        vvmultirend/vvtexmultirend.h \
        vvmultirend/vvtexmultirendmngr.h

libvirvo_a_SOURCES = \
        $(libvirvo_a_headers) \
        vvbrick.cpp \
        vvbsptree.cpp \
        vvcg.cpp \
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
        vvshadermanager.cpp \
        vvshaderfactory.cpp \
        vvrenderslave.cpp \
        vvrendertarget.cpp \
        vvrendercontext.cpp \
        vvoffscreenbuffer.cpp \
        vvmultirend/vvtexmultirend.cpp \
        vvmultirend/vvtexmultirendmngr.cpp

libvirvoincludedir = $(includedir)/virvo
libvirvoinclude_HEADERS = $(libvirvo_a_headers) config.h

INCLUDES=$(LIBVIRVO_CFLAGS) $(CG_INCLUDES) $(GLEW_INCLUDES) $(XLIBS_INCLUDES) $(STINGRAY_INCLUDES) $(PIC_FLAGS)
LIBS=$(LIBVIRVO_LIBS) $(CG_LIBS) $(GLEW_LIBS) $(GL_LIBS) $(XLIBS_LIBS) $(STINGRAY_LIBS)
