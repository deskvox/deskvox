uniform mat4 ModelProjectInv; // alte
//uniform mat4 ModelViewProjectionMatrix; // aktuell
uniform vec4 vp;

void main(void)
{
  vec4 pos;

  pos.x = (2.0 * (gl_Vertex.x - vp.x)) / vp.z - 1.0;
  pos.y = (2.0 * (gl_Vertex.y - vp.y)) / vp.w - 1.0;
  pos.z = (2.0 * gl_Vertex.z) - 1.0;

  pos.w = 1.0;

  vec4 obj = ModelProjectInv * pos;
  gl_Position = gl_ModelViewProjectionMatrix * obj;
  gl_FrontColor = gl_Color;
  gl_PointSize = 3.0 * pos.z;
}

