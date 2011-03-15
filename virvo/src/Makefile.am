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
        vvaabb.h \
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
        vvibrclient.h \
        vvibrserver.h \
        vvopengl.h \
        vvprintgl.h \
        vvpthread.h \
        vvrayrend.h \
        vvremoteclient.h \
        vvremoteserver.h \
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
	vvrendervp.h \
	vvsoftimg.h \
	vvsoftpar.h \
	vvsoftper.h \
	vvsoftvr.h \
	vvcuda.h \
        vvcudaimg.h \
        vvcudautils.h \
	vvcudasw.h \
	vvcudatransfunc.h \
        vvbonjour/vvbonjourbrowser.h \
        vvbonjour/vvbonjourentry.h \
        vvbonjour/vvbonjourregistrar.h \
        vvbonjour/vvbonjourresolver.h \
        vvmultirend/vvtexmultirend.h \
        vvmultirend/vvtexmultirendmngr.h

libvirvo_a_SOURCES = \
        $(libvirvo_a_headers) \
        vvaabb.cpp \
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
        vvibrclient.cpp \
        vvibrserver.cpp \
        vvprintgl.cpp \
        vvpthread.cpp \
        vvremoteclient.cpp \
        vvremoteserver.cpp \
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
        vvcudaimg.cpp \
        vvbonjour/vvbonjourbrowser.cpp \
        vvbonjour/vvbonjourentry.cpp \
        vvbonjour/vvbonjourregistrar.cpp \
        vvbonjour/vvbonjourresolver.cpp \
        vvmultirend/vvtexmultirend.cpp \
        vvmultirend/vvtexmultirendmngr.cpp

if VOLPACK
libvirvo_a_SOURCES += vvrendervp.cpp
endif

if CUDA
libvirvo_a_SOURCES += \
	vvcudasw.cu \
        vvrayrend.cu \
	vvcudatransfunc.cu
endif

libvirvoincludedir = $(includedir)/virvo
libvirvoinclude_HEADERS = $(libvirvo_a_headers) vvconfig.h

INCLUDES=$(LIBVIRVO_CFLAGS) $(CG_INCLUDES) $(GLEW_INCLUDES) $(XLIBS_INCLUDES) $(STINGRAY_INCLUDES) $(VOLPACK_INCLUDES) $(PIC_FLAGS) $(CUDA_INCLUDES)
LIBS=$(LIBVIRVO_LIBS) $(CG_LIBS) $(GLEW_LIBS) $(GL_LIBS) $(XLIBS_LIBS) $(STINGRAY_LIBS) $(VOLPACK_LIBS)

BUILT_SOURCES=vvversioninfo.h

vvversioninfo.h: FORCE
	$(srcdir)/../makeversion.sh $(srcdir) 0

FORCE:
