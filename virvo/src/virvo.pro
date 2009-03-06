!include($$(COVISEDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)
### don't modify anything before this line ###

TEMPLATE    = lib
                    
PROJECT     = General

CONFIG	   *= dll dl cg

SRCDIR = $${BASEDIR}/src/kernel/virvo

win32:DEFINES += VIRVO_EXPORT UNICODE
DEFINES *= NO_CONFIG_H

SOURCES	   = \
vvcolor.cpp vvdebugmsg.cpp vvdicom.cpp vvdynlib.cpp vvfileio.cpp vvgltools.cpp \
vvideo.cpp vvimage.cpp vvprintgl.cpp vvrenderer.cpp vvsocket.cpp vvsocketio.cpp \
vvsphere.cpp vvstingray.cpp vvstopwatch.cpp vvtexrend.cpp vvtfwidget.cpp \
vvtokenizer.cpp vvtoolshed.cpp vvtransfunc.cpp vvvecmath.cpp vvvffile.cpp \
vvvoldesc.cpp vvglsl.cpp \
vvmultirend/vvtexmultirend.cpp vvmultirend/vvtexmultirendmngr.cpp

HEADERS	   = \
glext-orig.h \
vvarray.h vvcolor.h vvdebugmsg.h vvdicom.h vvdynlib.h vvexport.h vvfileio.h vvglext.h \
vvgltools.h vvideo.h vvimage.h vvopengl.h vvprintgl.h vvrenderer.h vvsllist.h vvsocket.h \
vvsocketio.h vvsphere.h vvstingray.h vvstopwatch.h vvtexrend.h vvtfwidget.h vvtokenizer.h \
vvtoolshed.h vvtransfunc.h vvvecmath.h vvvffile.h vvvirvo.h vvvoldesc.h vvglsl.h \
vvmultirend/vvtexmultirend.h vvmultirend/vvtexmultirendmngr.h

TARGET		 = coVirvo

DEVFILES   = $$HEADERS

### don't modify anything below this line ###
!include ($$(COVISEDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)

# work around bug in intel compiler
caymanopt {
   QMAKE_CXXFLAGS -= -O2
   QMAKE_CXXFLAGS += -O1
   QMAKE_CXXFLAGS_RELEASE -= -O2
}
