AUTOMAKE_OPTIONS = foreign

if GLUT
noinst_PROGRAMS = vview
endif
vview_SOURCES = \
                vvview.cpp \
                vvobjview.cpp \
                vvperformancetest.cpp

LDADD = ../virvo/libvirvo.a

INCLUDES = -I.. -I$(srcdir)/..  $(CG_INCLUDES) $(GLEW_INCLUDES) $(STINGRAY_INCLUDES) $(VOLPACK_INCLUDES) $(GLUT_INCLUDES) $(GLU_INCLUDES) $(GL_INCLUDES) $(CUDA_FLAGS) $(PTHREAD_INCLUDES) $(XLIBS_INCLUDES) $(CUDA_INCLUDES)
LIBS = $(CONFIG_LIBS) $(CG_LIBS) $(GLEW_LIBS) $(STINGRAY_LIBS) $(VOLPACK_LIBS) $(GLUT_LIBS) $(GLU_LIBS) $(GL_LIBS) $(CUDA_LIBS) $(PTHREAD_LIBS) $(XLIBS_LIBS) $(LIBDL_LIBS)
