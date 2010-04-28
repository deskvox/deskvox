AUTOMAKE_OPTIONS = foreign

noinst_PROGRAMS = vview

vview_SOURCES = \
                vvview.cpp \
                vvobjview.cpp \
                vvperformancetest.cpp

INCLUDES = -I../src  $(CG_INCLUDES) $(GLEW_INCLUDES) $(STINGRAY_INCLUDES) $(GLUT_INCLUDES) $(GLU_INCLUDES) $(GL_INCLUDES) $(PTHREAD_INCLUDES) $(XLIBS_INCLUDES)
LIBS = -L../src -lvirvo $(CONFIG_LIBS) $(CG_LIBS) $(GLEW_LIBS) $(STINGRAY_LIBS) $(GLUT_LIBS) $(GLU_LIBS) $(GL_LIBS) $(PTHREAD_LIBS) $(XLIBS_LIBS) $(LIBDL_LIBS)
