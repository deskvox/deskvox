SET(CG_SHADERS
  vv_passthrough_frag_shader.cg
  vv_preint_frag.cg
  vv_preint_vert.cg
  vv_shader01.cg
  vv_shader02.cg
  vv_shader03.cg
  vv_shader04.cg
  vv_shader05.cg
  vv_shader06.cg
  vv_shader07.cg
  vv_shader08.cg
  vv_shader09.cg
  vv_shader10.cg
  vv_shader11.cg
  vv_shader12.cg
  vv_shader13.cg
)

SET(GLSL_SHADERS
  vv_intersection_geo.glsl
  vv_intersection_geo_inst.glsl
  vv_intersection.glsl
  vv_intersection_ver.glsl
  vv_intersection_ver_inst.glsl
)

IF(NOT COVISE_BUILD)
  INSTALL(FILES ${CG_SHADERS}
    DESTINATION ${INSTALL_DIR_SHARE}/virvo
  )

  INSTALL(FILES ${GLSL_SHADERS}
    DESTINATION ${INSTALL_DIR_SHARE}/virvo
  )
ENDIF(NOT COVISE_BUILD)
