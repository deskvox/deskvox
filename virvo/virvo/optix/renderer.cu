#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#undef MATH_NAMESPACE

#include <visionaray/math/aabb.h>
#include <visionaray/math/ray.h>

#include "framestate.h"
#include "volume.h"

#undef MATH_NAMESPACE

using namespace visionaray;

rtBuffer<FrameState, 1> frameStateBuffer;
rtBuffer<float4, 2>     colorBuffer;

// Volume frame(s)
rtBuffer<Volume, 1/*numframes*/> volumes;

// Transfer function(s)
rtBuffer<int, 1/*numtfs*/> transfuncs;

// Leaf buffer
rtBuffer<visionaray::aabb, 1> leafBuffer;

// The BVH
rtDeclareVariable(rtObject, volumeBVH, , );

// Grid
rtDeclareVariable(uint2, pixelID,   rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim,   );

// ------------------------------------------------------------------
// ray data
// ------------------------------------------------------------------
rtDeclareVariable(float, hit_t0, attribute hit_t0, );
rtDeclareVariable(float, hit_t1, attribute hit_t1, );
rtDeclareVariable(int,   hit_ID, attribute hit_ID, );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, optixRay, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance  , );

// Ray payload
struct VolumePRD {
  int leafID;
  float t0, t1;
};

rtDeclareVariable(VolumePRD,  volumePRD,   rtPayload, );

inline __device__ bool boxTest(const optix::Ray &ray, const aabb &box,
                               float &t0, float &t1)
{
  vec3 ori(optixRay.origin.x, optixRay.origin.y, optixRay.origin.z);
  vec3 dir(optixRay.direction.x, optixRay.direction.y, optixRay.direction.z);

  const vec3 t_lo = (box.min - ori) / dir;
  const vec3 t_hi = (box.max - ori) / dir;

  const vec3 t_nr = min(t_lo,t_hi);
  const vec3 t_fr = max(t_lo,t_hi);

  t0 = max(ray.tmin,max_element(t_nr));
  t1 = min(ray.tmax,min_element(t_fr));

  return t0 < t1;
}

// Ray generation program
RT_PROGRAM void renderFrame()
{
  colorBuffer[pixelID] = make_float4(0,0,0,0);

  const matrix_camera& cam = frameStateBuffer[0].camera;

  float delta = frameStateBuffer[0].delta;

  ray r = cam.primary_ray(
        ray{},
        (float)pixelID.x,
        (float)pixelID.y,
        (float)launchDim.x,
        (float)launchDim.y
        );

  optix::Ray optixRay = optix::Ray(make_float3(r.ori.x,r.ori.y,r.ori.z),
                                   make_float3(r.dir.x,r.dir.y,r.dir.z),
                                   0 /* ray type */,
                                   0.f, /* tmin */
                                   2e10f /*tmax */);

  float alreadyIntegratedDistance = 0.f;

  while (1) {

    VolumePRD prd;
    prd.leafID = -1;
    prd.t0 = prd.t1 = 0.f; // doesn't matter as long as leafID==-1
    optixRay.ray_type = 0;//BRICK_RAY_TYPE;
    optixRay.tmin = alreadyIntegratedDistance;
    optixRay.tmax = 1e20f;//surface.t_hit * dt_scale;
    rtTrace(volumeBVH, optixRay, prd,
            RT_VISIBILITY_ALL,
            RT_RAY_FLAG_DISABLE_ANYHIT
            );
    if (prd.leafID < 0)
      break;

    alreadyIntegratedDistance = prd.t1 * (1.0000001f);

    float dt = delta;
    float t = prd.t0;
    float tmax = prd.t1;
    float4& dst = colorBuffer[pixelID];
    visionaray::aabb bbox = volumes[0].bbox;
    float3 size = make_float3(bbox.size().x,bbox.size().y,bbox.size().z);

    // integrate
    float3 pos = optixRay.origin + optixRay.direction * t;

    float3 tex_coord = make_float3(
            ( pos.x + (size.x / 2) ) / size.x,
            (-pos.y + (size.y / 2) ) / size.y,
            (-pos.z + (size.z / 2) ) / size.z
            );

    float3 inc = (optixRay.direction * delta / size) * make_float3(1,-1,-1);

    while (t < tmax)
    {
      float voxel = optix::rtTex3D<float>(volumes[0].texID, tex_coord.x, tex_coord.y, tex_coord.z);
      float4 color = optix::rtTex1D<float4>(transfuncs[0], voxel);

      // opacity correction
      color.w = 1.0f - pow(1.0f - color.w, dt);

      // premultiplied alpha
      color = make_float4(color.x * color.w,
                          color.y * color.w,
                          color.z * color.w,
                          color.w);

      // compositing
      dst += color * (1.0f - dst.w);

      // step on
      tex_coord += inc;
      t += dt;
    }
  }
}

// Bounding box program
RT_PROGRAM void getBounds(int leafID, float result[6])
{
  result[0] = leafBuffer[leafID].min.x;
  result[1] = leafBuffer[leafID].min.y;
  result[2] = leafBuffer[leafID].min.z;
  result[3] = leafBuffer[leafID].max.x;
  result[4] = leafBuffer[leafID].max.y;
  result[5] = leafBuffer[leafID].max.z;

  printf("%f %f %f  %f %f %f\n",
         result[0],
         result[1],
         result[2],
         result[3],
         result[4],
         result[5]);
}

// Intersection program
RT_PROGRAM void intersection(int leafID)
{
    float t0 = optixRay.tmin, t1 = optixRay.tmax;
    const aabb &brick = leafBuffer[leafID];
    if (!boxTest(optixRay,brick,t0,t1))
      return;

    if (rtPotentialIntersection(t0)) {
      hit_t0 = t0;
      hit_t1 = t1;
      hit_ID = leafID;
      rtReportIntersection(0);
    }
}

// Closest hit program
RT_PROGRAM void closestHit()
{
  volumePRD.t0 = hit_t0;
  volumePRD.t1 = hit_t1;
  volumePRD.leafID = hit_ID;
}
