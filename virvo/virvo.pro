# compile with 'qmake -makefile'

TEMPLATE = lib
#QT = core
CONFIG += staticlib
DESTDIR = bin
OBJECTS_DIR = obj
MOC_DIR = obj
UI_DIR = obj
#LIBS += -Llib -lGLEW
INCLUDEPATH += src
DEFINES += HAVE_CG

SOURCES += src/vvcolor.cpp \
        src/vvdebugmsg.cpp \
        src/vvdicom.cpp \
        src/vvdynlib.cpp \
        src/vvfileio.cpp \
        src/vvgltools.cpp \
        src/vvimage.cpp \
        src/vvprintgl.cpp \
        src/vvrenderer.cpp \
        src/vvsocket.cpp \
        src/vvsocketio.cpp \
        src/vvsphere.cpp \
        src/vvstingray.cpp \
        src/vvstopwatch.cpp \
        src/vvtexrend.cpp \
        src/vvtfwidget.cpp \
        src/vvtokenizer.cpp \
        src/vvtoolshed.cpp \
        src/vvtransfunc.cpp \
        src/vvvecmath.cpp \
        src/vvvffile.cpp \
        src/vvvoldesc.cpp \
        src/vvglsl.cpp \
        src/vvmultirend/vvtexmultirend.cpp \
        src/vvmultirend/vvtexmultirendmngr.cpp

HEADERS += src/glext-orig.h \
        src/vvarray.h \
        src/vvcolor.h \
        src/vvdebugmsg.h \
        src/vvdicom.h \
        src/vvdynlib.h \
        src/vvexport.h \
        src/vvfileio.h \
        src/vvglext.h \
        src/vvgltools.h \
        src/vvimage.h \
        src/vvopengl.h \
        src/vvprintgl.h \
        src/vvrenderer.h \
        src/vvsllist.h \
        src/vvsocket.h \
        src/vvsocketio.h \
        src/vvsphere.h \
        src/vvstingray.h \
        src/vvstopwatch.h \
        src/vvtexrend.h \
        src/vvtfwidget.h \
        src/vvtokenizer.h \
        src/vvtoolshed.h \
        src/vvtransfunc.h \
        src/vvvecmath.h \
        src/vvvffile.h \
        src/vvvirvo.h \
        src/vvvoldesc.h \
        src/vvglsl.h \
        src/vvmultirend/vvtexmultirend.h \
        src/vvmultirend/vvtexmultirendmngr.h


