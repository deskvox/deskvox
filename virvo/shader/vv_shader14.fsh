// Shader for up to 4 channels with independent TFs

// Author: Martin Aumueller <aumueller@hlrs.de>

uniform int channels;

uniform sampler3D pix3dtex;

uniform sampler2D pixLUT0;
uniform sampler2D pixLUT1;
uniform sampler2D pixLUT2;
uniform sampler2D pixLUT3;

void main()
{
  vec4 data = texture3D(pix3dtex, gl_TexCoord[0].xyz); // data from texture for up to 4 channels

  vec4 c[4];
  c[0] = texture2D(pixLUT0, vec2(data.x, 0.0));
  float maxAlpha = c[0].a;
  if (channels == 2)
  {
     c[1] = texture2D(pixLUT1, vec2(data.w, 0.0));
     maxAlpha = max(maxAlpha, c[1].a);
  }
  else if (channels >= 3)
  {
     c[1] = texture2D(pixLUT1, vec2(data.y, 0.0));
     maxAlpha = max(maxAlpha, c[1].a);
     c[2] = texture2D(pixLUT2, vec2(data.z, 0.0));
     maxAlpha = max(maxAlpha, c[2].a);
  }
  if (channels >= 4)
  {
     c[3] = texture2D(pixLUT3, vec2(data.w, 0.0));
     maxAlpha = max(maxAlpha, c[3].a);
  }

  c[0].rgb *= c[0].a;
  for (int i=1; i<channels; ++i)
  {
    c[0].rgb += c[i].rgb * c[i].a;
  }
  c[0].rgb /= maxAlpha;

  gl_FragColor.rgb = c[0].rgb;
  gl_FragColor.a = maxAlpha;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
