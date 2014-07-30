// Shader for up to 4 channels with independent TFs

// Author: Martin Aumueller <aumueller@hlrs.de>

#define DELTA (0.01)

uniform int channels;
uniform int preintegration;
uniform int lighting;

uniform vec3 V;
uniform vec3 lpos;
uniform float constAtt;
uniform float linearAtt;
uniform float quadAtt;
uniform float threshold;

uniform sampler3D pix3dtex;

uniform sampler2D pixLUT0;
uniform sampler2D pixLUT1;
uniform sampler2D pixLUT2;
uniform sampler2D pixLUT3;

vec4 classify(sampler2D lut, float s0, float s1, bool preint)
{
  if (preint)
    return texture2D(lut, vec2(s0, s1));
  else
    return texture2D(lut, vec2(s0, 0.0));
}

vec3 gradient(sampler3D tex, vec3 tc)
{
    vec3 sample1;
    vec3 sample2;

    sample1.x = texture3D(tex, tc + vec3(DELTA, 0.0, 0.0)).x;
    sample2.x = texture3D(tex, tc - vec3(DELTA, 0.0, 0.0)).x;
    // signs for y and z are swapped because of texture orientation
    sample1.y = texture3D(tex, tc - vec3(0.0, DELTA, 0.0)).x;
    sample2.y = texture3D(tex, tc + vec3(0.0, DELTA, 0.0)).x;
    sample1.z = texture3D(tex, tc - vec3(0.0, 0.0, DELTA)).x;
    sample2.z = texture3D(tex, tc + vec3(0.0, 0.0, DELTA)).x;

    return sample2.xyz - sample1.xyz;
}

vec4 light(sampler3D tex, vec4 classified, vec3 tc)
{
  const vec3 Ka = vec3(0.3, 0.3, 0.3);
  const vec3 Kd = vec3(0.8, 0.8, 0.8);
  const vec3 Ks = vec3(0.8, 0.8, 0.8);
  const float shininess = 1000.0;

  if (lighting!=0 && classified.w > threshold)
  {
    vec3 grad = gradient(tex, tc);
    vec3 L = lpos - tc;
    float dist = length(L);
    L /= dist;
    vec3 N = normalize(grad);
    vec3 H = normalize(L + V);

    float att = 1.0 / (constAtt + linearAtt * dist + quadAtt * dist * dist);
    float ldot = dot(L, N.xyz);
    float specular = pow(dot(H, N.xyz), shininess);

    // Ambient term.
    vec3 col = Ka * classified.xyz;

    // Diffuse term.
    col += Kd * ldot * classified.xyz * att;

    // Specular term.
    float spec = pow(dot(H, N), shininess);
    col += Ks * spec * classified.xyz * att;

    return vec4(col, classified.w);
  }
  else
    return classified;
}

void main()
{
  bool preint = preintegration==0 ? false : true;
  vec4 data = texture3D(pix3dtex, gl_TexCoord[0].xyz); // data from texture for up to 4 channels
  vec3 tc = gl_TexCoord[0].xyz;
  vec4 data1;
  if (preint)
  {
    data1 = texture3D(pix3dtex, gl_TexCoord[1].xyz);
    tc += gl_TexCoord[1].xyz;
    tc *= 0.5;
  }
  else
  {
    data1 = vec4(0., 0., 0., 0.);
  }

  vec4 c[4];
  c[0] = classify(pixLUT0, data.x, data1.x, preint);
  c[0] = light(pix3dtex, c[0], tc);
  float maxAlpha = c[0].a;
  if (channels == 2)
  {
    c[1] = classify(pixLUT1, data.w, data1.w, preint);
    c[1] = light(pix3dtex, c[1], tc);
    maxAlpha = max(maxAlpha, c[1].a);
  }
  else if (channels >= 3)
  {
    c[1] = classify(pixLUT1, data.y, data1.y, preint);
    c[1] = light(pix3dtex, c[1], tc);
    maxAlpha = max(maxAlpha, c[1].a);
    c[2] = classify(pixLUT2, data.z, data1.z, preint);
    c[2] = light(pix3dtex, c[2], tc);
    maxAlpha = max(maxAlpha, c[2].a);
  }
  if (channels >= 4)
  {
    c[3] = classify(pixLUT2, data.w, data1.w, preint);
    c[3] = light(pix3dtex, c[3], tc);
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
