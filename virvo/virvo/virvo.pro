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
SOURCES = \
  vvbonjour/vvbonjourbrowser.cpp \
  vvbonjour/vvbonjourentry.cpp \
  vvbonjour/vvbonjourregistrar.cpp \
  vvbonjour/vvbonjourresolver.cpp \
  vvmultirend/vvtexmultirend.cpp \
  vvmultirend/vvtexmultirendmngr.cpp \
  vvaabb.cpp \
  vvbrick.cpp \
  vvbsptree.cpp \
  vvbsptreevisitors.cpp \
  vvcgprogram.cpp \
  #vvclusterclient.cpp \
  #vvclusterserver.cpp \
  vvclock.cpp \
  vvcolor.cpp \
  vvcuda.cpp \
  vvcudaimg.cpp \
  vvcudatools.cpp \
  vvdebugmsg.cpp \
  vvdicom.cpp \
  vvdynlib.cpp \
  vvfileio.cpp \
  vvglslprogram.cpp \
  vvgltools.cpp \
  vvibr.cpp \
  vvibrclient.cpp \
  vvibrimage.cpp \
  vvibrserver.cpp \
  vvvideo.cpp \
  vvimage.cpp \
  vvimageclient.cpp \
  vvimageserver.cpp \
  vvmulticast.cpp \
  vvoffscreenbuffer.cpp \
  vvprintgl.cpp \
  vvpthread.cpp \
  vvremoteclient.cpp \
  vvremoteserver.cpp \
  vvrendercontext.cpp \
  vvrenderer.cpp \
  vvrendererfactory.cpp \
  vvrendertarget.cpp \
  vvrendervp.cpp \
  vvshaderfactory.cpp \
  vvshaderprogram.cpp \
  vvsocket.cpp \
  vvsocketio.cpp \
  vvsocketmonitor.cpp \
  vvsoftimg.cpp \
  vvsoftpar.cpp \
  vvsoftper.cpp \
  vvsoftvr.cpp \
  vvsoftsw.cpp \
  vvsphere.cpp \
  vvstingray.cpp \
  vvtexrend.cpp \
  vvtfwidget.cpp \
  vvtokenizer.cpp \
  vvtoolshed.cpp \
  vvtransfunc.cpp \
  vvvecmath.cpp \
  vvvffile.cpp \
  vvvirvo.cpp \
  vvvisitor.cpp \
  vvvoldesc.cpp 
HEADERS = \
  vvbonjour/vvbonjourbrowser.h \
  vvbonjour/vvbonjourentry.h \
  vvbonjour/vvbonjourregistrar.h \
  vvbonjour/vvbonjourresolver.h \
  vvmultirend/vvtexmultirend.h \
  vvmultirend/vvtexmultirendmngr.h \
  vvaabb.h \
  vvbrick.h \
  vvbsptree.h \
  vvbsptreevisitors.h \
  vvcgprogram.h \
  #vvclusterclient.h \
  #vvclusterserver.h \
  vvclock.h \
  vvcomparisonrend.h \
  vvcomparisonrend.impl.h \
  vvcolor.h \
  vvcuda.h \
  vvcudaimg.h \
  vvcudasw.h \
  vvcudatools.h \
  vvcudatransfunc.h \
  vvcudautils.h \
  vvdebugmsg.h \
  vvdicom.h \
  vvdynlib.h \
  vvexport.h \
  vvfileio.h \
  vvglew.h \
  vvglslprogram.h \
  vvgltools.h \
  vvibr.h \
  vvibrclient.h \
  vvibrimage.h \
  vvibrserver.h \
  vvvideo.h \
  vvimage.h \
  vvimageclient.h \
  vvimageserver.h \
  vvinttypes.h \
  vvmulticast.h \
  vvoffscreenbuffer.h \
  vvopengl.h \
  vvparam.h \
  vvplatform.h \
  vvprintgl.h \
  vvpthread.h \
  vvrayrend.h \
  vvremoteclient.h \
  vvremoteserver.h \
  vvrendercontext.h \
  vvrenderer.h \
  vvrendererfactory.h \
  vvrendertarget.h \
  vvrendervp.h \
  vvshaderfactory.h \
  vvshaderprogram.h \
  vvswitchrenderer.h \
  vvswitchrenderer.impl.h \
  vvsllist.h \
  vvsocket.h \
  vvsocketio.h \
  vvsocketmonitor.h \
  vvsoftimg.h \
  vvsoftpar.h \
  vvsoftper.h \
  vvsoftvr.h \
  vvsoftsw.h \
  vvsphere.h \
  vvstingray.h \
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
  vvx11.h 
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
