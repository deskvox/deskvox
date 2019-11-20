#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <GL/glew.h>

#include "cuda/timer.h"

#include <optixu/optixpp_namespace.h>

#undef MATH_NAMESPACE

#include <visionaray/math/aabb.h>
#include <visionaray/math/rectangle.h>

#include "optix/framestate.h"
#include "optix/volume.h"

#undef MATH_NAMESPACE

#include "private/vvgltools.h"
#include "vvcudarendertarget.h"
#include "vvoptixrenderer.h"
#include "vvspaceskip.h"
#include "vvtextureutil.h"
#include "vvvoldesc.h"

using namespace visionaray;

#define TF_WIDTH 256

extern "C" const char ptxCode[];

struct vvOptixRenderer::Impl
{
  Impl() : tree(virvo::SkipTree::SVTKdTree) {}

  virvo::SkipTree tree;

  // General optix
  optix::Context          context;
  optix::Program          raygenProgram;

  // Buffers with specific frame state
  optix::Buffer           colorBuffer;
  optix::Buffer           frameStateBuffer;

  // BVH
  optix::Buffer           leafBuffer;
  optix::GeometryGroup    volumeGG;
  optix::GeometryInstance volumeGI;
  optix::Geometry         volumeG;
  optix::Material         volumeMat;

  // 3-d volume texture
  optix::Buffer           volumeBuffer;
  optix::TextureSampler   volume;
  optix::Buffer           volumeIDs;

  // Transfer function texture
  optix::Buffer           transfuncBuffer;
  optix::TextureSampler   transfunc;
  optix::Buffer           transfuncIDs;

  void initOptix()
  {
    context = optix::Context::create();
    context->setRayTypeCount(1);
    context->setEntryPointCount(1);

    // Frame state buffer
    frameStateBuffer = context->createBuffer(RT_BUFFER_INPUT);
    frameStateBuffer->setFormat(RT_FORMAT_USER);
    frameStateBuffer->setElementSize(sizeof(FrameState));
    frameStateBuffer->setSize(1);
    context["frameStateBuffer"]->set(frameStateBuffer);

    // Color buffer
    colorBuffer = context->createBuffer(RT_BUFFER_INPUT|RT_BUFFER_OUTPUT);
    colorBuffer->setFormat(RT_FORMAT_FLOAT4);
    colorBuffer->setSize(1,1); // this will get resized later on, anyway
    context["colorBuffer"]->set(colorBuffer);

    // Ray generation program
    raygenProgram = context->createProgramFromPTXString(ptxCode, "renderFrame");
    context->setRayGenerationProgram(0, raygenProgram);

    const int RTX = true;
    if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(RTX), &RTX) != RT_SUCCESS)
      printf("Error setting RTX mode. \n");
    else
      printf("OptiX RTX execution mode is %s.\n", (RTX) ? "on" : "off");
  }

  void createBVH(const std::vector<visionaray::aabb>& leaves)
  {
    volumeG = context->createGeometry();
    volumeG->setPrimitiveCount((int)leaves.size());

    // Leaf buffer
    leafBuffer = context->createBuffer(RT_BUFFER_INPUT);
    leafBuffer->setFormat(RT_FORMAT_USER);
    leafBuffer->setElementSize(sizeof(visionaray::aabb)); //sf->value.size());
    leafBuffer->setSize(leaves.size()); //sf->value.size());

    visionaray::aabb* mappedLeaves = (visionaray::aabb*)leafBuffer->map();
    memcpy(mappedLeaves, leaves.data(), leaves.size() * sizeof(visionaray::aabb));
    leafBuffer->unmap();
    context["leafBuffer"]->set(leafBuffer);

    // BVH
    optix::Program bbProgram
      = context->createProgramFromPTXString(ptxCode,"getBounds");
    volumeG->setBoundingBoxProgram(bbProgram);
    optix::Program isecProgram
      = context->createProgramFromPTXString(ptxCode,"intersection");
    volumeG->setIntersectionProgram(isecProgram);
      
    volumeMat = context->createMaterial();
    optix::Program chProgram
      = context->createProgramFromPTXString(ptxCode,"closestHit");
    volumeMat->setClosestHitProgram(/*BRICK_RAY_TYPE*/0,chProgram);
    // optix::Program ahProgram
    //   = context->createProgramFromPTXString(ptxCode,"exa::Brick_any_hit");
    // volumeMat->setAnyHitProgram(BRICK_RAY_TYPE,ahProgram);
      
    volumeGI = context->createGeometryInstance();
    volumeGI->setGeometry(volumeG);
    volumeGI->setMaterialCount(1);
    volumeGI->setMaterial(0,volumeMat);
      
    // -------------------------------------------------------
    // create dummy buffers and world so createSurfaces can do launches
    // -------------------------------------------------------
    volumeGG = context->createGeometryGroup();
    volumeGG->setChildCount(1);
    volumeGG->setChild(0,volumeGI);
    volumeGG->setAcceleration(context->createAcceleration("Bvh"));
    context["volumeBVH"]->set(volumeGG);
  }

  void createVolumeTexture(uint8_t* data, int width, int height, int depth,
                           const visionaray::aabb& worldBounds,
                           RTfiltermode filterMode)
  {
    volumeBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, width, height, depth);
    uint8_t* buf = (uint8_t*)volumeBuffer->map();
    memcpy(buf, data, width*height*depth*sizeof(uint8_t));
    volumeBuffer->unmap();

    volume = context->createTextureSampler();
    volume->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    volume->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    volume->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    volume->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    volume->setFilteringModes(filterMode, filterMode, RT_FILTER_NONE);
    volume->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT/*_SRGB*/);
    volume->setBuffer(volumeBuffer);

    volumeIDs = context->createBuffer(RT_BUFFER_INPUT);
    volumeIDs->setFormat(RT_FORMAT_USER);
    volumeIDs->setElementSize(sizeof(Volume));
    volumeIDs->setSize(1/*num frames */);
    Volume* volumes = (Volume*)volumeIDs->map();
    volumes[0] = {volume->getId(),worldBounds};
    volumeIDs->unmap();
    context["volumes"]->set(volumeIDs);
  }

  void createTransfuncTexture(vec4* data, int size)
  {
    transfuncBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, size);
    vec4* buf = (vec4*)transfuncBuffer->map();
    memcpy(buf, data, size*sizeof(vec4));
    transfuncBuffer->unmap();

    transfunc = context->createTextureSampler();
    transfunc->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    transfunc->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    //transfunc->setFilteringModes(RT_FILTER_NONE,RT_FILTER_LINEAR,RT_FILTER_NONE);
    transfunc->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    transfunc->setBuffer(transfuncBuffer);

    transfuncIDs = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, 1);
    int* ids = (int*)transfuncIDs->map();
    ids[0] = transfunc->getId();
    transfuncIDs->unmap();
    context["transfuncs"]->set(transfuncIDs);
  }
};

vvOptixRenderer::vvOptixRenderer(vvVolDesc* vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
  , impl_(new Impl)
{
  rendererType = RAYRENDOPTIX;

  glewInit();

  virvo::RenderTarget* rt = virvo::HostBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);
  setRenderTarget(rt);

  impl_->initOptix();

  updateVolumeData();
  updateTransferFunction();
}

vvOptixRenderer::~vvOptixRenderer()
{
}

void vvOptixRenderer::renderVolumeGL()
{
  mat4 view_matrix;
  mat4 proj_matrix;
  recti viewport;

  glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix.data());
  glGetFloatv(GL_PROJECTION_MATRIX, proj_matrix.data());
  glGetIntegerv(GL_VIEWPORT, viewport.data());

  // determine ray integration step size (aka delta)
  int axis = 0;
  if (vd->getSize()[1] / vd->vox[1] < vd->getSize()[axis] / vd->vox[axis])
  {
      axis = 1;
  }
  if (vd->getSize()[2] / vd->vox[2] < vd->getSize()[axis] / vd->vox[axis])
  {
      axis = 2;
  }

  float delta = (vd->getSize()[axis] / vd->vox[axis]) / _quality;

  matrix_camera cam(view_matrix, proj_matrix);

  cam.begin_frame();

  FrameState *fs = (FrameState *)impl_->frameStateBuffer->map();
  fs->camera = cam;
  fs->delta = delta;
  impl_->frameStateBuffer->unmap();

  cam.end_frame();

  impl_->colorBuffer->setSize(viewport.w, viewport.h);

  virvo::CudaTimer t;
  impl_->context->launch(0, viewport.w, viewport.h);

  virvo::RenderTarget* rt = getRenderTarget();
  const float* colors = (const float*)impl_->colorBuffer->map();
  memcpy((void*)rt->deviceColor(), colors, viewport.w * viewport.h * 4 * sizeof(float));

  impl_->colorBuffer->unmap();

  if (_boundaries)
  {
    virvo::vec4 clearColor = vvGLTools::queryClearColor();
    vvColor color(1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2]);

    impl_->tree.renderGL(color);
  }

  std::cout << std::fixed << std::setprecision(8);
  static double avg = 0.0;
  static size_t cnt = 0;
  avg += t.elapsed();
  cnt += 1;
  std::cout << avg/cnt << std::endl;
}

void vvOptixRenderer::updateTransferFunction()
{
  std::vector<vec4> tf(TF_WIDTH * 1 * 1);
  vd->computeTFTexture(0, TF_WIDTH, 1, 1, reinterpret_cast<float*>(tf.data()));

  impl_->tree.updateTransfunc(reinterpret_cast<const uint8_t*>(tf.data()), TF_WIDTH, 1, 1, virvo::PF_RGBA32F);

  virvo::vec3 eye(getEyePosition().x, getEyePosition().y, getEyePosition().z);
  bool frontToBack = true;
  auto bricks = impl_->tree.getSortedBricks(eye, frontToBack);

  std::vector<visionaray::aabb> visionarayBricks(bricks.size()); // with alignment

  for (size_t i = 0; i < bricks.size(); ++i)
  {
    visionarayBricks[i] = visionaray::aabb(
            visionaray::vec3(bricks[i].min.x,bricks[i].min.y,bricks[i].min.z),
            visionaray::vec3(bricks[i].max.x,bricks[i].max.y,bricks[i].max.z)
            );
  }

  impl_->createBVH(visionarayBricks);

  impl_->createTransfuncTexture((vec4*)tf.data(), tf.size());
}

void vvOptixRenderer::updateVolumeData()
{
  vvRenderer::updateVolumeData();

  impl_->tree.updateVolume(*vd);

  virvo::PixelFormat texture_format = virvo::PF_R8;

  virvo::TextureUtil tu(vd);
  //for (int f = 0; f < vd->frames; ++f) 
  int f = 0;
  {
    virvo::TextureUtil::Pointer tex_data = nullptr;

    tex_data = tu.getTexture(virvo::vec3i(0),
            virvo::vec3i(vd->vox),
            texture_format,
            virvo::TextureUtil::All,
            f);  

    RTfiltermode filterMode = getParameter(VV_SLICEINT).asInt() == virvo::Linear ?
                                            RT_FILTER_LINEAR :
                                            RT_FILTER_NONE;

    virvo::aabb bbox = vd->getBoundingBox();
    visionaray::aabb worldBounds({bbox.min.x,bbox.min.y,bbox.min.z},
                                 {bbox.max.x,bbox.max.y,bbox.max.z});
    impl_->createVolumeTexture((uint8_t*)tex_data, vd->vox[0], vd->vox[1], vd->vox[2],
                               worldBounds, filterMode);
  }
}

void vvOptixRenderer::setCurrentFrame(size_t frame)
{
}

bool vvOptixRenderer::checkParameter(ParameterType param, vvParam const& value) const
{
}

void vvOptixRenderer::setParameter(ParameterType param, const vvParam& newValue)
{
  vvRenderer::setParameter(param, newValue);
}

bool vvOptixRenderer::instantClassification() const
{
  return true;
}
