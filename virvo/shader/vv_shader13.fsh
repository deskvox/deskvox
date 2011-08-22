// Author of cg-version: 
//  Stefan Zellmann <zellmans@uni-koeln.de>
//
// Converted by:
//  Stavros Delisavas <stavros.delisavas@uni-koeln.de>

#define DELTA (0.01)
#define THRESHOLD (0.1)

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;
uniform vec3 L;
uniform vec3 H;

void main()
{
  // TODO: make these parameters uniform and configurable.
  const vec3 Ka = vec3(0.0, 0.0, 0.0);
  const vec3 Kd = vec3(0.8, 0.8, 0.8);
  const vec3 Ks = vec3(0.8, 0.8, 0.8);
  const float shininess = 1000.0;

  vec4 center;
  float centerwx = texture3D(pix3dtex, gl_TexCoord[0].xyz).x;
  center.w = centerwx;
  center.x = centerwx;
  vec4 classification = texture2D(pixLUT, center.wx);

  vec4 OUT;
  if (classification.w > THRESHOLD)
  {
    vec3 N;
    vec3 sample1;
    vec3 sample2;
    sample1.x = texture3D(pix3dtex, gl_TexCoord[0].xyz - vec3(DELTA, 0.0, 0.0)).x;
    sample2.x = texture3D(pix3dtex, gl_TexCoord[0].xyz + vec3(DELTA, 0.0, 0.0)).x;
    sample1.y = texture3D(pix3dtex, gl_TexCoord[0].xyz - vec3(0.0, DELTA, 0.0)).x;
    sample2.y = texture3D(pix3dtex, gl_TexCoord[0].xyz + vec3(0.0, DELTA, 0.0)).x;
    sample1.z = texture3D(pix3dtex, gl_TexCoord[0].xyz - vec3(0.0, 0.0, DELTA)).x;
    sample2.z = texture3D(pix3dtex, gl_TexCoord[0].xyz + vec3(0.0, 0.0, DELTA)).x;

    N = normalize(sample2.xyz - sample1.xyz);

    float ldot = dot(L, N.xyz);
    float specular = pow(dot(H, N.xyz), shininess);

    // Ambient term.
    OUT.xyz = Ka * classification.xyz;

    if (ldot > 0.0)
    {
      // Diffuse term.
      OUT.xyz += Kd * ldot * classification.xyz;

      // Specular term.
      float spec = pow(dot(H, N), shininess);

      if (spec > 0.0)
      {
        OUT.xyz += Ks * spec * classification.xyz;
      }
    }
    OUT.w = classification.w;
    gl_FragColor = OUT;
  }
  else
  {
    gl_FragColor = classification;
  }
}
