
TEMPLATE    = lib
                    
PROJECT     = General

CONFIG	   *= dll dl cg

win32:DEFINES += VIRVO_EXPORTS
DEFINES *= NO_CONFIG_H

SOURCES	   = \
vvcolor.cpp vvdebugmsg.cpp vvdicom.cpp vvdynlib.cpp vvfileio.cpp vvgltools.cpp vvideo.cpp \
vvimage.cpp vvprintgl.cpp vvrenderer.cpp vvsocket.cpp vvsocketio.cpp vvsphere.cpp vvstingray.cpp \
vvstopwatch.cpp vvtexrend.cpp vvtfwidget.cpp vvtokenizer.cpp vvtoolshed.cpp vvtransfunc.cpp \
vvvecmath.cpp vvvffile.cpp vvvoldesc.cpp

HEADERS	   = \
glext-orig.h vvarray.h vvcolor.h vvdebugmsg.h vvdicom.h vvdynlib.h vvexport.h vvfileio.h vvglext.h \
vvgltools.h vvideo.h vvimage.h vvopengl.h vvprintgl.h vvrenderer.h vvsllist.h vvsocket.h \
vvsocketio.h vvsphere.h vvstingray.h vvstopwatch.h vvtexrend.h vvtfwidget.h vvtokenizer.h \
vvtoolshed.h vvtransfunc.h vvvecmath.h vvvffile.h vvvirvo.h vvvoldesc.h

TARGET		 = coVirvo

DEVFILES   = $$HEADERS

!include ($$BASEDIR/mkspecs/config.pro) {
    message (include of config.pro failed)
}
