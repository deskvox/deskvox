uniform mat4 reprojectionMatrix;
uniform sampler2D rgbaTex;
uniform sampler2D depthTex;
uniform float imageWidth;
uniform float imageHeight;
uniform float vpWidth;
uniform float vpHeight;
uniform float splitX, splitY;
uniform float depthMin;
uniform float depthRange;
uniform bool closer;

void main(void)
{
  float x = gl_Vertex.x;
  float y = gl_Vertex.y;
  if(closer)
  {
    // start at projection center
    if(gl_Vertex.x < splitX)
      x = floor(splitX)-gl_Vertex.x;
    if(gl_Vertex.y < splitY)
      y = floor(splitY)-gl_Vertex.y;
  }
  else
  {
    // proceed outside-in
    if(gl_Vertex.x > splitX)
      x = imageWidth+floor(splitX)-gl_Vertex.x;
    if(gl_Vertex.y > splitY)
      y = imageHeight+floor(splitY)-gl_Vertex.y;
  }
  vec2 tc = vec2(x/imageWidth, y/imageHeight);
  gl_FrontColor = texture2D(rgbaTex, tc);
  vec4 c = gl_FrontColor;
  float d = texture2D(depthTex, tc).r;
  // p will be in normalized device coordinates
  vec4 p = vec4(tc.x, tc.y, depthMin+d*depthRange, 1.);
  p.xyz *= 2.;
  p.xyz -= vec3(1., 1., 1.);
  gl_Position = reprojectionMatrix * p;
  if(d <= 0.)
  {
    gl_Position.z = gl_Position.w * 1.1; // clip vertex
  }
#if 0
  else
  {
    p += vec4(1./imageWidth, 1./imageHeight, 0., 0.);
    vec4 s1 = gl_Position;
    vec4 s2 = reprojectionMatrix * p;
    s1 /= s1.w;
    s1 *= vpWidth/imageWidth;
    s2 /= s2.w;
    s2 *= vpHeight/imageHeight;
    vec2 d = vec2((s1.x-s2.x), (s1.y-s2.y));
    gl_PointSize = vpWidth * length(d) * 0.7; // 0.7 = 1/sqrt(2)
  }
#endif
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
