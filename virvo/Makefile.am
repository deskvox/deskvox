SUBDIRS = src test
AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = makeversion.sh

SHADERS = \
	shader/vv_intersection.glsl \
	shader/vv_intersection_ver.glsl \
	shader/vv_intersection_geo.glsl \
	shader/vv_shader01.cg \
	shader/vv_shader02.cg \
	shader/vv_shader03.cg \
	shader/vv_shader04.cg \
	shader/vv_shader05.cg \
	shader/vv_shader06.cg \
	shader/vv_shader07.cg \
	shader/vv_shader08.cg \
	shader/vv_shader09.cg \
	shader/vv_shader10.cg \
	shader/vv_shader11.cg \
	shader/vv_shader12.cg \
	shader/vv_shader13.cg

shaderdatadir = $(datadir)/virvo
dist_shaderdata_DATA = $(SHADERS)
