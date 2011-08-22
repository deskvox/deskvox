// Shader for pre-integrated rendering of scalar data
//
// Author of cg-version: 
//  Martin Aumueller <aumueller@uni-koeln.de>
//
// Converted by:
//  Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;

void main()
{
  float x = texture3D(pix3dtex, gl_TexCoord[0].xyz).x;;
  float y = texture3D(pix3dtex, gl_TexCoord[1].xyz).x;;
  vec4 OUT = texture2D(pixLUT, vec2(x, y));

  gl_FragColor = OUT;
}
