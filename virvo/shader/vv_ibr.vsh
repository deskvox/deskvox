uniform mat4 reprojectionMatrix;
uniform sampler2D rgbaTex;
uniform sampler2D depthTex;
uniform float imageWidth;
uniform float imageHeight;
uniform float vpWidth;
uniform float vpHeight;

void main(void)
{
  vec2 tc = vec2(gl_Vertex.x/imageWidth, gl_Vertex.y/imageHeight);
  gl_FrontColor = texture2D(rgbaTex, tc);
  vec4 p = gl_Vertex;
  p.z = texture2D(depthTex, tc).r;
  gl_Position = reprojectionMatrix * p;
  if(p.z <= 0.)
  {
    gl_Position.z = gl_Position.w * 1.1;
  }
  else
  {
    p += vec4(vpWidth/imageWidth, vpHeight/imageHeight, 0., 0.);
    vec4 s1 = gl_Position;
    vec4 s2 = reprojectionMatrix * p;
    s1 /= s1.w;
    s2 /= s2.w;
    vec2 d = vec2((s1.x-s2.x)*vpWidth, (s1.y-s2.y)*vpHeight);
    gl_PointSize = length(d) * 0.7; // 0.7 = 1/sqrt(2)
  }
}

