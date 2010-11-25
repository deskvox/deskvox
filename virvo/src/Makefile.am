AUTOMAKE_OPTIONS = foreign

EXTRA_DIST =

DISTCLEANFILES =

.cu.lo:
	$(top_builddir)/cudalt.py $@ $(NVCC) $(DEFS) -c $(NVCCFLAGS) $<
.cu.o:
	$(NVCC) $(DEFS) -c $(NVCCFLAGS) $<

lib_LIBRARIES = libvirvo.a

libvirvo_a_headers = \
        vvversioninfo.h \
        vvarray.h \
        vvbrick.h \
        vvbsptree.h \
        vvbsptreevisitors.h \
        vvcg.h \
        vvcolor.h \
        vvdebugmsg.h \
        vvdicom.h \
        vvdynlib.h \
        vvexport.h \
        vvfileio.h \
        vvglew.h \
        vvgltools.h \
        vvimage.h \
        vvopengl.h \
        vvprintgl.h \
        vvpthread.h \
        vvrayrend.h \
        vvrenderer.h \
        vvsllist.h \
        vvsocket.h \
        vvsocketio.h \
        vvsocketmonitor.h \
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
        vvrendermaster.h \
        vvrenderslave.h \
        vvrendertarget.h \
        vvrendercontext.h \
        vvoffscreenbuffer.h \
        vvx11.h \
	vvsoftimg.h \
	vvsoftpar.h \
	vvsoftper.h \
	vvsoftvr.h \
	vvcuda.h \
        vvcudautils.h \
	vvcudapar.h \
        vvbonjour/vvbonjourbrowser.h \
        vvbonjour/vvbonjourentry.h \
        vvbonjour/vvbonjourregistrar.h \
        vvbonjour/vvbonjourresolver.h \
        vvmultirend/vvtexmultirend.h \
        vvmultirend/vvtexmultirendmngr.h

libvirvo_a_SOURCES = \
        $(libvirvo_a_headers) \
        vvbrick.cpp \
        vvbsptree.cpp \
        vvbsptreevisitors.cpp \
        vvcg.cpp \
        vvcolor.cpp \
        vvdebugmsg.cpp \
        vvdicom.cpp \
        vvdynlib.cpp \
        vvfileio.cpp \
        vvgltools.cpp \
        vvimage.cpp \
        vvprintgl.cpp \
        vvpthread.cpp \
        vvrayrend.cu \
        vvrenderer.cpp \
        vvsocket.cpp \
        vvsocketio.cpp \
        vvsocketmonitor.cpp \
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
        vvvirvo.cpp \
        vvvoldesc.cpp \
        vvglsl.cpp \
        vvshadermanager.cpp \
        vvshaderfactory.cpp \
        vvrendermaster.cpp \
        vvrenderslave.cpp \
        vvrendertarget.cpp \
        vvrendercontext.cpp \
        vvoffscreenbuffer.cpp \
	vvsoftimg.cpp \
	vvsoftpar.cpp \
	vvsoftper.cpp \
	vvsoftvr.cpp \
	vvcuda.cpp \
	vvcudapar.cu \
        vvbonjour/vvbonjourbrowser.cpp \
        vvbonjour/vvbonjourentry.cpp \
        vvbonjour/vvbonjourregistrar.cpp \
        vvbonjour/vvbonjourresolver.cpp \
        vvmultirend/vvtexmultirend.cpp \
        vvmultirend/vvtexmultirendmngr.cpp

libvirvoincludedir = $(includedir)/virvo
libvirvoinclude_HEADERS = $(libvirvo_a_headers) vvconfig.h

INCLUDES=$(LIBVIRVO_CFLAGS) $(CG_INCLUDES) $(GLEW_INCLUDES) $(XLIBS_INCLUDES) $(STINGRAY_INCLUDES) $(PIC_FLAGS) $(CUDA_INCLUDES)
LIBS=$(LIBVIRVO_LIBS) $(CG_LIBS) $(GLEW_LIBS) $(GL_LIBS) $(XLIBS_LIBS) $(STINGRAY_LIBS)

BUILT_SOURCES=vvversioninfo.h

vvversioninfo.h: FORCE
	$(abs_top_srcdir)/makeversion.sh $(abs_top_srcdir) $(abs_srcdir) 0

FORCE:
