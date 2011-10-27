!include($$(COVISEDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)

# ## don't modify anything before this line ###
TEMPLATE = lib
PROJECT = General
CONFIG *= dll \
    dl \
    cg \
    gdcm \
    glew \
    pthread
CONFIG *= wnoerror
unix:!macx:CONFIG *= x11
osx11:CONFIG *= x11
SRCDIR = $${BASEDIR}/src/kernel/virvo
win32:DEFINES += VIRVO_EXPORT \
    UNICODE \
    NOMINMAX
DEFINES *= NO_CONFIG_H
SOURCES = vvaabb.cpp \
    vvbrick.cpp \
    vvcolor.cpp \
    vvdebugmsg.cpp \
    vvcgprogram.cpp \
    vvcuda.cpp \
    vvcudaimg.cpp \
    vvdicom.cpp \
    vvdynlib.cpp \
    vvfileio.cpp \
    vvglslprogram.cpp \
    vvgltools.cpp \
    vvvideo.cpp \
    vvimage.cpp \
    vvibrimage.cpp \
    vvoffscreenbuffer.cpp \
    vvprintgl.cpp \
    vvpthread.cpp \
    vvremoteclient.cpp \
    vvrendercontext.cpp \
    vvrenderer.cpp \
    vvrendererfactory.cpp \
    vvrendertarget.cpp \
    vvsocket.cpp \
    vvsocketio.cpp \
    vvsocketmonitor.cpp \
    vvsphere.cpp \
    vvstingray.cpp \
    vvclock.cpp \
    vvtexrend.cpp \
    vvtfwidget.cpp \
    vvtokenizer.cpp \
    vvtoolshed.cpp \
    vvtransfunc.cpp \
    vvvecmath.cpp \
    vvvisitor.cpp \
    vvvffile.cpp \
    vvvirvo.cpp \
    vvvoldesc.cpp \
    vvshaderfactory.cpp \
    vvshaderprogram.cpp \
    vvbsptree.cpp \
    vvbsptreevisitors.cpp \
    vvsoftimg.cpp \
    vvsoftpar.cpp \
    vvsoftper.cpp \
    vvsoftvr.cpp \
    vvsoftsw.cpp \
    vvbonjour/vvbonjourbrowser.cpp \
    vvbonjour/vvbonjourentry.cpp \
    vvbonjour/vvbonjourregistrar.cpp \
    vvbonjour/vvbonjourresolver.cpp \
    vvmultirend/vvtexmultirend.cpp \
    vvmultirend/vvtexmultirendmngr.cpp \
    vvibr.cpp \
    vvibrclient.cpp \
    vvibrserver.cpp \
    vvimageclient.cpp \
    vvimageserver.cpp \
    vvremoteserver.cpp
HEADERS = vvaabb.h \
    vvarray.h \
    vvbrick.h \
    vvcgprogram.h \
    vvcolor.h \
    vvcudatransfunc.h \
    vvdebugmsg.h \
    vvcuda.h \
    vvcudaimg.h \
    vvcudautils.h \
    vvdicom.h \
    vvdynlib.h \
    vvexport.h \
    vvfileio.h \
    vvglslprogram.h \
    vvgltools.h \
    vvvideo.h \
    vvimage.h \
    vvibrimage.h \
    vvoffscreenbuffer.h \
    vvopengl.h \
    vvglew.h \
    vvprintgl.h \
    vvpthread.h \
    vvrayrend.h \
    vvremoteclient.h \
    vvrendercontext.h \
    vvrenderer.h \
    vvrendererfactory.h \
    vvrendertarget.h \
    vvsllist.h \
    vvsocket.h \
    vvsocketio.h \
    vvsocketmonitor.h \
    vvsphere.h \
    vvstingray.h \
    vvclock.h \
    vvtexrend.h \
    vvtfwidget.h \
    vvtokenizer.h \
    vvtoolshed.h \
    vvtransfunc.h \
    vvvecmath.h \
    vvvffile.h \
    vvvirvo.h \
    vvvisitor.h \
    vvvoldesc.h \
    vvshaderfactory.h \
    vvshaderprogram.h \
    vvbsptree.h \
    vvx11.h \
    vvbsptreevisitors.h \
    vvsoftimg.h \
    vvsoftpar.h \
    vvsoftper.h \
    vvsoftvr.h \
    vvsoftsw.h \
    vvbonjour/vvbonjourbrowser.h \
    vvbonjour/vvbonjourentry.h \
    vvbonjour/vvbonjourregistrar.h \
    vvbonjour/vvbonjourresolver.h \
    vvmultirend/vvtexmultirend.h \
    vvmultirend/vvtexmultirendmngr.h \
    vvibr.h \
    vvibrclient.h \
    vvibrserver.h \
    vvimageclient.h \
    vvimageserver.h \
    vvremoteserver.h
TARGET = coVirvo
DEVFILES = $$HEADERS
DEFINES += VV_COVISE

CUDA = $$(CUDA_DEFINES)
contains(CUDA,HAVE_CUDA) {
        CONFIG *= cuda \
                  cudart
        CUDA_SOURCES += vvcudatransfunc.cu \
                        vvcudasw.cu \
                        vvrayrend.cu
        DEFINES *= HAVE_CUDA
}
QMAKE_CUFLAGS += "-D__builtin_stdarg_start=__builtin_va_start"

# ## don't modify anything below this line ###
!include ($$(COVISEDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)

# work around bug in intel compiler - keep after config-last.pri
caymanopt {
    QMAKE_CXXFLAGS -= -O2
    QMAKE_CXXFLAGS += -O1
    QMAKE_CXXFLAGS_RELEASE -= -O2
}
