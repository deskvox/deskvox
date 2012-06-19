SET(CG_SHADERS
  vv_preint_frag.cg
  vv_preint_vert.cg
  vv_shader01.frag.cg
  vv_shader02.frag.cg
  vv_shader03.frag.cg
  vv_shader04.frag.cg
  vv_shader05.frag.cg
  vv_shader06.frag.cg
  vv_shader07.frag.cg
  vv_shader08.frag.cg
  vv_shader09.frag.cg
  vv_shader10.frag.cg
  vv_shader11.frag.cg
  vv_shader12.frag.cg
  vv_shader13.frag.cg
)

SET(GLSL_SHADERS
  vv_ibr.fsh
  vv_ibr.vsh
  vv_isect_geom_only.gsh
  vv_isect_geom_only.vsh
  vv_isect_vert_geom_combined.gsh
  vv_isect_vert_geom_combined.vsh
  vv_isect_vert_only.vsh
  vv_preint_frag.cg
  vv_preint_vert.cg
  vv_shader01.fsh
  vv_shader02.fsh
  vv_shader03.fsh
  vv_shader04.fsh
  vv_shader05.fsh
  vv_shader06.fsh
  vv_shader07.fsh
  vv_shader08.fsh
  vv_shader09.fsh
  vv_shader10.fsh
  vv_shader11.fsh
  vv_shader12.fsh
  vv_shader13.fsh
)

IF(NOT COVISE_BUILD)
  INSTALL(FILES ${CG_SHADERS}
    DESTINATION ${INSTALL_DIR_SHARE}/virvo
  )

  INSTALL(FILES ${GLSL_SHADERS}
    DESTINATION ${INSTALL_DIR_SHARE}/virvo
  )
ENDIF(NOT COVISE_BUILD)
