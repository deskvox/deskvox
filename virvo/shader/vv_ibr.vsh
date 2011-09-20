uniform mat4 reprojectionMatrix;

void main(void)
{
  gl_Position = reprojectionMatrix * gl_Vertex;
  gl_FrontColor = gl_Color;
  //gl_PointSize = 3.0 * pos.z;
}

