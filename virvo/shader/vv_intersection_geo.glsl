#version 120
#extension GL_EXT_geometry_shader4 : enable

uniform vec3 planeNormal;

uniform float delta;
uniform int sequence[64];
uniform vec3 vertices[8];
uniform vec4 brickMin;
uniform vec4 brickDimInv;
uniform vec3 texMin;
uniform vec3 texRange;
uniform int v1Maybe[3];
uniform int v2Maybe[3];

varying in float planeDist[3];

bool test(const int idx)
{
  int vIdx1 = sequence[int(brickDimInv[3]) + v1Maybe[idx]];
  int vIdx2 = sequence[int(brickDimInv[3]) + v2Maybe[idx]];
    
  vec3 vecV1 = vertices[vIdx1];
  vec3 vecV2 = vertices[vIdx2];

  vec3 vecStart = vecV1;
  vec3 vecDir = vecV2-vecV1;

  float denominator = dot(vecDir, planeNormal);
  float lambda = (denominator != 0.0) ?
                    (planeDist[0]-dot(vecStart, planeNormal)) / denominator
                  :
                    -1.0;

  if ((lambda >= 0.0) && (lambda <= 1.0))
  {
    vec3 pos = vecStart + lambda * vecDir;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0);
    gl_TexCoord[0].xyz = (pos - brickMin.xyz) * brickDimInv.xyz;
    gl_TexCoord[0].xyz = gl_TexCoord[0].xyz * texRange + texMin;
    EmitVertex();
    return true;
  }
  else
  {
    return false;
  }
}

void send(const int idx)
{
  gl_Position = gl_PositionIn[idx];
  gl_TexCoord[0] = gl_TexCoordIn[idx][0];
  EmitVertex();
}

void main()
{
  test(0); // p1
  send(0); // p0
  send(1); // p2
  if (test(2)) // p5
  {
    test(1); // p3
    send(2); // p4
  }
  else
  {
    send(2); // p4
    test(1); // p3
  }
  EndPrimitive();
}
