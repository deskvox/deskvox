
#ifndef NDEBUG
#include <iostream>
#include <ostream>
#endif
#include <vector>

#include <GL/glew.h>

#include <thrust/device_vector.h>

#undef MATH_NAMESPACE

#include <visionaray/bvh.h>
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/material.h>
#include <visionaray/pixel_format.h>
#include <visionaray/pixel_traits.h>
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/result_record.h>
#include <visionaray/scheduler.h>
#include <visionaray/shade_record.h>
#include <visionaray/detail/stack.h>

#undef MATH_NAMESPACE

#include "cuda/utils.h"
#include "gl/util.h"
#include "vvclock.h"
#include "vvcudarendertarget.h"
#include "vvsimplecaster.h"
#include "vvspaceskip.h"
#include "vvtextureutil.h"
#include "vvvoldesc.h"

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
//
//

template <typename T, typename Tex> 
VSNRAY_FUNC 
inline vector<3, T> gradient(Tex const& tex, vector<3, T> tex_coord) 
{ 
    vector<3, T> s1; 
    vector<3, T> s2; 
 
    float DELTA = 0.01f; 
 
    s1.x = tex3D(tex, tex_coord + vector<3, T>(DELTA, 0.0f, 0.0f)); 
    s2.x = tex3D(tex, tex_coord - vector<3, T>(DELTA, 0.0f, 0.0f)); 
    // signs for y and z are swapped because of texture orientation 
    s1.y = tex3D(tex, tex_coord - vector<3, T>(0.0f, DELTA, 0.0f)); 
    s2.y = tex3D(tex, tex_coord + vector<3, T>(0.0f, DELTA, 0.0f)); 
    s1.z = tex3D(tex, tex_coord - vector<3, T>(0.0f, 0.0f, DELTA)); 
    s2.z = tex3D(tex, tex_coord + vector<3, T>(0.0f, 0.0f, DELTA)); 
 
    return s2 - s1; 
}

namespace visionaray
{
VSNRAY_FUNC
inline bool is_closer(hit_record<ray, aabb> query, hit_record<ray, aabb> reference)
{
    return query.tnear < reference.tnear;
}
}


//-------------------------------------------------------------------------------------------------
// Volume rendering kernel
//

struct Kernel
{
    template <typename C>
    VSNRAY_FUNC
    C shade(C color, vec3 view_dir, vec3 pos) const
    {
        if (color.w < 0.1f)
            return color;

        auto grad = gradient(volume, pos);

        if (length(grad) == 0.0f)
            return color;

        plastic<float> mat; 
        mat.ca() = from_rgb(vector<3, float>(0.3f, 0.3f, 0.3f));
        mat.cd() = from_rgb(vector<3, float>(0.8f, 0.8f, 0.8f));
        mat.cs() = from_rgb(vector<3, float>(0.8f, 0.8f, 0.8f));
        mat.ka() = 1.0f;
        mat.kd() = 1.0f;
        mat.ks() = 1.0f;
        mat.specular_exp() = 12.0f;

        shade_record<float> sr;
        sr.normal = normalize(grad);
        sr.geometric_normal = sr.normal;
        sr.view_dir = view_dir;
        sr.tex_color = vec3(1.0f);
        sr.light_dir = normalize(light.position());
        sr.light_intensity = light.intensity(pos);

        color.xyz() = color.xyz() * to_rgb(mat.shade(sr));

        return color;
    }

    template <typename R, typename T, typename C>
    VSNRAY_FUNC
    void integrate(R ray, T t, T tmax, C& dst) const
    {
        integrate(ray, t, tmax, dst, delta);
    }

    template <typename R, typename T, typename C>
    VSNRAY_FUNC
    void integrate(R ray, T t, T tmax, C& dst, T dt) const
    {
        while (t < tmax)
        {
            auto pos = ray.ori + ray.dir * t;
            vector<3, T> tex_coord(
                    ( pos.x + (bbox.size().x / 2) ) / bbox.size().x,
                    (-pos.y + (bbox.size().y / 2) ) / bbox.size().y,
                    (-pos.z + (bbox.size().z / 2) ) / bbox.size().z
                    );

            T voxel = tex3D(volume, tex_coord);
            C color = tex1D(transfunc, voxel);

            color = shade(color, -ray.dir, tex_coord);

            // opacity correction
            color.w = 1.0f - pow(1.0f - color.w, dt);

            // premultiplied alpha
            color.xyz() *= color.w;

            // compositing
            dst += color * (1.0f - dst.w);

            // step on
            t += dt;
        }
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> operator()(R ray) const
    {
        using S = typename R::scalar_type;
        using C = vector<4, S>;

        result_record<S> result;
        result.color = C(0.0);

        auto hit_rec = intersect(ray, bbox);
        result.hit = hit_rec.hit;

        if (!hit_rec.hit)
            return result;

        auto t = max(S(0.0f), hit_rec.tnear);
        auto tmax = hit_rec.tfar;

#if 0//KDTREE
        for (int i = 0; i < num_kd_tree_leaves; ++i)
        {
            auto kd_hit_rec = intersect(ray, kd_tree_leaves[i]);

            if (kd_hit_rec.hit)
            {
                auto kd_t = max(t, kd_hit_rec.tnear);
                auto kd_tmax = min(tmax, kd_hit_rec.tfar);

                integrate(ray, kd_t, kd_tmax, result.color);

                t = kd_tmax;
            }
        }
#endif
//        using HR = hit_record<R, aabb>; //decltype(hit_rec);
//        HR current;
        visionaray::hit_record<visionaray::basic_ray<float>, visionaray::basic_aabb<float>> current;
        current.tnear = t;
        current.tfar  = tmax;
        current.hit   = true;

        // traverse tree
        detail::stack<32> st;
        st.push(0);

        auto inv_dir = 1.0f / ray.dir;

next:
        while(!st.empty())
        {
            auto node = nodes[st.pop()];

            while (!node.is_leaf())
            {
                auto children = &nodes[node.first_child];

                auto hr1 = intersect(ray, children[0].get_bounds(), inv_dir);
                auto hr2 = intersect(ray, children[1].get_bounds(), inv_dir);

                auto b1 = visionaray::any( is_closer(hr1, current) );
                auto b2 = visionaray::any( is_closer(hr2, current) );

                if (b1 && b2)
                {
                    unsigned near_addr = visionaray::all( hr1.tnear < hr2.tnear ) ? 0 : 1;
                    st.push(node.get_child(!near_addr));
                    node = children[near_addr];
                }
                else if (b1)
                {
                    node = children[0];
                }
                else if (b2)
                {
                    node = children[1];
                }
                else
                {
                    goto next;
                }
            }

            // traverse leaf
            auto hr = intersect(ray, node.get_bounds(), inv_dir);
            integrate(ray, max(current.tnear, hr.tnear), hr.tfar, result.color);
            current.tnear = hr.tfar;
        }

//#else
//    integrate(ray, t, tmax, result.color);
//#endif

        return result;
    }

    cuda_texture<unorm<8>, 3>::ref_type volume;
    cuda_texture<vec4, 1>::ref_type transfunc;

    aabb bbox;
    vec3i vox;
    float delta;
    bool local_shading;
    point_light<float> light;

    visionaray::bvh_node* nodes = nullptr;

};

class virvo_render_target
{
public:

    static const pixel_format CF = PF_RGBA32F;
    static const pixel_format DF = PF_UNSPECIFIED;

    using color_type = typename pixel_traits<CF>::type;
    using depth_type = typename pixel_traits<DF>::type;

    using ref_type = render_target_ref<CF, DF>;

public:

    virvo_render_target(int w, int h, color_type* c, depth_type* d)
        : width_(w)
        , height_(h)
        , color_(c)
        , depth_(d)
    {
    }

    int width() const { return width_; }
    int height() const { return height_; }

    color_type* color() { return color_; }
    depth_type* depth() { return depth_; }

    color_type const* color() const { return color_; }
    depth_type const* depth() const { return depth_; }

    ref_type ref() { return { color(), depth(), width(), height() }; }

    void begin_frame() {}
    void end_frame() {}

    int width_;
    int height_;

    color_type* color_;
    depth_type* depth_;
};

struct vvSimpleCaster::Impl
{
    Impl() : sched(8, 8)
    , tree(virvo::SkipTree::SVTKdTree)
    {}

    using R = basic_ray<float>;

    cuda_sched<R> sched;

    std::vector<cuda_texture<unorm<8>, 3>> volumes;

    cuda_texture<vec4, 1> transfunc;

    virvo::SkipTree tree;

    thrust::device_vector<bvh_node> device_tree;
};

vvSimpleCaster::vvSimpleCaster(vvVolDesc* vd, vvRenderState renderState)
: vvRenderer(vd, renderState), impl_(new Impl)
{
    rendererType = RAYRENDSIMPLE;

    glewInit();

    virvo::cuda::initGlInterop();

    virvo::RenderTarget* rt = virvo::PixelUnpackBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);

    // no direct rendering
    if (rt == NULL)
    {
        rt = virvo::DeviceBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);
    }
    setRenderTarget(rt);

    updateVolumeData();
    updateTransferFunction();
}

vvSimpleCaster::~vvSimpleCaster()
{
}

void vvSimpleCaster::renderVolumeGL()
{
    mat4 view_matrix;
    mat4 proj_matrix;
    recti viewport;

    glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix.data());
    glGetFloatv(GL_PROJECTION_MATRIX, proj_matrix.data());
    glGetIntegerv(GL_VIEWPORT, viewport.data());

    virvo::RenderTarget* rt = getRenderTarget();

    assert(rt);

    virvo_render_target virvo_rt(
        rt->width(),
        rt->height(),
        static_cast<virvo_render_target::color_type*>(rt->deviceColor()),
        static_cast<virvo_render_target::depth_type*>(rt->deviceDepth())
        );

    auto sparams = make_sched_params(
        view_matrix,
        proj_matrix,
        virvo_rt
        );

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

    auto bbox = vd->getBoundingBox();

    // Lights
    point_light<float> light;

    if (getParameter(VV_LIGHTING))
    {
        assert( glIsEnabled(GL_LIGHTING) );
        auto l = virvo::gl::getLight(GL_LIGHT0);
        vec4 lpos(l.position.x, l.position.y, l.position.z, l.position.w);

        light.set_position( (inverse(view_matrix) * lpos).xyz() );
        light.set_cl(vec3(l.diffuse.x, l.diffuse.y, l.diffuse.z));
        light.set_kl(l.diffuse.w);
        light.set_constant_attenuation(l.constant_attenuation);
        light.set_linear_attenuation(l.linear_attenuation);
        light.set_quadratic_attenuation(l.quadratic_attenuation);
    }

    Kernel kernel;

    kernel.volume = cuda_texture<unorm<8>, 3>::ref_type(impl_->volumes[vd->getCurrentFrame()]);
    kernel.transfunc = cuda_texture<vec4, 1>::ref_type(impl_->transfunc);

    kernel.bbox          = aabb(vec3(bbox.min.data()), vec3(bbox.max.data()));
    kernel.vox           = vec3i(vd->vox[0], vd->vox[1], vd->vox[2]);
    kernel.delta         = delta;
    kernel.local_shading = getParameter(VV_LIGHTING);
    kernel.light         = light;

    kernel.nodes         = thrust::raw_pointer_cast(impl_->device_tree.data());
//
//    glDisable(GL_DEPTH_TEST);
//
//#if KDTREE
//    vec3 eye(getEyePosition().x, getEyePosition().y, getEyePosition().z);
//    auto leaves = impl_->kdtree.get_leaf_nodes(eye);
//    thrust::device_vector<aabb> d_leaves(leaves);
//    kernel.kd_tree_leaves = thrust::raw_pointer_cast(d_leaves.data());
//    kernel.num_kd_tree_leaves = static_cast<int>(d_leaves.size());
//#endif
//
//#if FRAME_TIMING
//    cudaEvent_t start;
//    cudaEvent_t stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);
//#endif
//
    impl_->sched.frame(kernel, sparams);
//
//#if FRAME_TIMING
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float ms = 0.0f;
//    cudaEventElapsedTime(&ms, start, stop);
//    std::cout << std::fixed << std::setprecision(3) << "Elapsed time: " << ms << "ms\n";
//#endif
}

void vvSimpleCaster::updateTransferFunction()
{
    std::vector<vec4> tf(256 * 1 * 1);
    vd->computeTFTexture(0, 256, 1, 1, reinterpret_cast<float*>(tf.data()));

    impl_->transfunc = cuda_texture<vec4, 1>(tf.size());
    impl_->transfunc.reset(tf.data());
    impl_->transfunc.set_address_mode(Clamp);
    impl_->transfunc.set_filter_mode(Nearest);

    texture_ref<vec4, 1> tf_ref(tf.size());
    tf_ref.reset(tf.data());
    tf_ref.set_address_mode(Clamp);
    tf_ref.set_filter_mode(Nearest);

    //impl_->tree.updateTransfunc(reinterpret_cast<const uint8_t*>(tf_ref.data()), 256, 1, 1, virvo::PF_RGBA32F);
    //auto data = impl_->tree.getPacked();
    //impl_->device_tree.resize(data.size());
    //bvh_node* tmp = reinterpret_cast<bvh_node*>(data.data());
    //thrust::copy(tmp, tmp + data.size(), impl_->device_tree.begin());
}

void vvSimpleCaster::updateVolumeData()
{
    vvRenderer::updateVolumeData();

    impl_->tree.updateVolume(*vd);


    // Init GPU textures
    tex_filter_mode filter_mode = getParameter(VV_SLICEINT).asInt() == virvo::Linear ? Linear : Nearest;

    virvo::PixelFormat texture_format = virvo::PF_R8;

    impl_->volumes.resize(vd->frames);


    virvo::TextureUtil tu(vd);
    for (int f = 0; f < vd->frames; ++f)
    {
        virvo::TextureUtil::Pointer tex_data = nullptr;

        tex_data = tu.getTexture(virvo::vec3i(0),
            virvo::vec3i(vd->vox),
            texture_format,
            virvo::TextureUtil::All,
            f);

        impl_->volumes[f] = cuda_texture<unorm<8>, 3>(vd->vox[0], vd->vox[1], vd->vox[2]);
        impl_->volumes[f].reset(reinterpret_cast<unorm<8> const*>(tex_data));
        impl_->volumes[f].set_address_mode(Clamp);
        impl_->volumes[f].set_filter_mode(filter_mode);
    }
}

void  vvSimpleCaster::setCurrentFrame(size_t frame) 
{
}

bool vvSimpleCaster::checkParameter(ParameterType param, vvParam const& value) const 
{
    return false;
}

void vvSimpleCaster::setParameter(ParameterType param, const vvParam& value) 
{
    switch (param)
    {
    case VV_SLICEINT:
        {
            if (_interpolation != static_cast< virvo::tex_filter_mode >(value.asInt()))
            {
                _interpolation = static_cast< virvo::tex_filter_mode >(value.asInt());
                tex_filter_mode filter_mode = _interpolation == virvo::Linear ? Linear : Nearest;

                for (auto& tex : impl_->volumes)
                {
                    tex.set_filter_mode(filter_mode);
                }
            }
        }
        break;

    default:
        vvRenderer::setParameter(param, value);
        break;
    }
}

bool vvSimpleCaster::instantClassification() const 
{
    return true;
}

