!include($$(COVISEDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)

# ## don't modify anything before this line ###
TEMPLATE = lib
PROJECT = General
CONFIG *= dll \
    dl \
    cg \
    gdcm \
    glew \
    pthread \
    wnoerror
unix:!macx:CONFIG *= x11
osx11:CONFIG *= x11
SRCDIR = $${BASEDIR}/src/kernel/virvo
win32:DEFINES += VIRVO_EXPORT \
    UNICODE
DEFINES *= NO_CONFIG_H
SOURCES = vvbrick.cpp \
    vvcolor.cpp \
    vvdebugmsg.cpp \
    vvdicom.cpp \
    vvdynlib.cpp \
    vvfileio.cpp \
    vvgltools.cpp \
    vvideo.cpp \
    vvimage.cpp \
    vvoffscreenbuffer.cpp \
    vvprintgl.cpp \
    vvrendercontext.cpp \
    vvrenderer.cpp \
    vvrendermaster.cpp \
    vvrenderslave.cpp \
    vvrendertarget.cpp \
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
    vvvisitor.cpp \
    vvvffile.cpp \
    vvvirvo.cpp \
    vvvoldesc.cpp \
    vvshaderfactory.cpp \
    vvshadermanager.cpp \
    vvcg.cpp \
    vvglsl.cpp \
    vvbsptree.cpp \
    vvbsptreevisitors.cpp \
    vvmultirend/vvtexmultirend.cpp \
    vvmultirend/vvtexmultirendmngr.cpp
HEADERS = \
    vvarray.h \
    vvbrick.h \
    vvcolor.h \
    vvdebugmsg.h \
    vvdicom.h \
    vvdynlib.h \
    vvexport.h \
    vvfileio.h \
    vvgltools.h \
    vvideo.h \
    vvimage.h \
    vvoffscreenbuffer.h \
    vvopengl.h \
    vvglew.h \
    vvprintgl.h \
    vvrendercontext.h \
    vvrenderer.h \
    vvrendermaster.h \
    vvrenderslave.h \
    vvrendertarget.h \
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
    vvvisitor.h \
    vvvoldesc.h \
    vvshaderfactory.h \
    vvshadermanager.h \
    vvcg.h \
    vvglsl.h \
    vvbsptree.h \
    vvbsptreevisitors.h \
    vvmultirend/vvtexmultirend.h \
    vvmultirend/vvtexmultirendmngr.h
TARGET = coVirvo
DEVFILES = $$HEADERS
DEFINES += VV_COVISE

# ## don't modify anything below this line ###
!include ($$(COVISEDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)

# work around bug in intel compiler
caymanopt { 
    QMAKE_CXXFLAGS -= -O2
    QMAKE_CXXFLAGS += -O1
    QMAKE_CXXFLAGS_RELEASE -= -O2
}
