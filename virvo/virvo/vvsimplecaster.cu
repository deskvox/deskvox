
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
#include "private/vvgltools.h"
#include "vvclock.h"
#include "vvcudarendertarget.h"
#include "vvsimplecaster.h"
#include "vvspaceskip.h"
#include "vvtextureutil.h"
#include "vvvoldesc.h"

#include "cuda/timer.h"
#include <iomanip>

using namespace visionaray;

#define TF_WIDTH 256

#define SIMPLE_GRID 0


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
        auto pos = ray.ori + ray.dir * t;

        vector<3, T> tex_coord(
                ( pos.x + (bbox.size().x / 2) ) / bbox.size().x,
                (-pos.y + (bbox.size().y / 2) ) / bbox.size().y,
                (-pos.z + (bbox.size().z / 2) ) / bbox.size().z
                );

        vector<3, T> inc = (ray.dir * delta / bbox.size()) * vec3(1,-1,-1);

        while (t < tmax)
        {
            T voxel = tex3D(volume, tex_coord);
            C color = tex1D(transfunc, voxel);

            //color = shade(color, -ray.dir, tex_coord);

            // opacity correction
            color.w = 1.0f - pow(1.0f - color.w, dt);

            // premultiplied alpha
            color.xyz() *= color.w;

            // compositing
            dst += color * (1.0f - dst.w);

            // step on
            tex_coord += inc;
            t += dt;
        }
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> ray_marching_naive(R ray) const
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

        integrate(ray, t, tmax, result.color);

        return result;
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> ray_marching_traverse_grid(R ray) const
    {
        using S = typename R::scalar_type;
        using C = vector<4, S>;

        result_record<S> result;
        result.color = C(0.0);

        auto hit_rec = intersect(ray, bbox);
        result.hit = hit_rec.hit;

        if (!hit_rec.hit)
            return result;

#if !defined(SIMPLE_GRID)
#if defined(RANGE_TREE) && __CUDA_ARCH__ > 0
        __shared__ float max_opacities[TF_WIDTH];
        int index = (threadIdx.y * blockDim.x + threadIdx.x) * 4; // 8x8 blocks..
        if (index < TF_WIDTH) {
            max_opacities[index] = max_opacities_[index];
            max_opacities[index+1] = max_opacities_[index+1];
            max_opacities[index+2] = max_opacities_[index+2];
            max_opacities[index+3] = max_opacities_[index+3];
        }
        __syncthreads();
#else
        float const* max_opacities = max_opacities_;
#endif
#endif

        auto t = max(S(0.0f), hit_rec.tnear);
        auto tmax = hit_rec.tfar;

        // cf. GridAccelerator_stepRay

        // Tentatively advance the ray.
        t += delta;

        const vec3f ray_rdir = 1.0f / ray.dir;
        // sign of direction determines near/far index
        const vec3i nextCellIndex = vec3i(1 - (*reinterpret_cast<unsigned*>(&ray.dir.x) >> 31),
                                          1 - (*reinterpret_cast<unsigned*>(&ray.dir.y) >> 31),
                                          1 - (*reinterpret_cast<unsigned*>(&ray.dir.z) >> 31));

        vec3i hit_cell;
        while (t < tmax)
        {
            // Compute the hit point in the local coordinate system.
            vec3f pos = ray.ori + t * ray.dir;
            vector<3, S> coord01(
                    (pos.x + (bbox.size().x / 2)) / bbox.size().x,
                    (pos.y + (bbox.size().y / 2)) / bbox.size().y,
                    (pos.z + (bbox.size().z / 2)) / bbox.size().z
                    );
            vec3f localCoordinates = coord01 * vec3(vox-1);

            // Compute the 3D index of the cell containing the hit point.
            vec3i cellIndex = vec3i(localCoordinates) >> 4;//>> CELL_WIDTH_BITCOUNT;

            // If we visited this cell before then it must not be empty.
            if (cellIndex == hit_cell)
            {
                integrate(ray, t, t + delta, result.color);
                t += delta;
                continue;
            }

            // Track the hit cell.
            hit_cell = cellIndex;

#ifdef SIMPLE_GRID
            uint8_t empty = cells_empty[(grid_dims.z-cellIndex.z-1) * grid_dims.x * grid_dims.y + (grid_dims.y-cellIndex.y-1) * grid_dims.x + cellIndex.x];
            float maximumOpacity = empty ? .0f : 1.f;
#else
            // Get the volumetric value range of the cell.
            vec2f cellRange = cell_ranges[(grid_dims.z-cellIndex.z-1) * grid_dims.x * grid_dims.y + (grid_dims.y-cellIndex.y-1) * grid_dims.x + cellIndex.x];

            // Get the maximum opacity in the volumetric value range.
            int tf_width = TF_WIDTH;
            int rx = floor(cellRange.x * 255.0f);
            int ry = ceil(cellRange.y * 255.0f);
#ifdef RANGE_TREE
            float maximumOpacity = 0;
            {
                int lo = rx;
                int hi = ry-1;
                if ((lo & 1) == 1) maximumOpacity = max(maximumOpacity, max_opacities[lo]);
                if ((hi & 1) == 0) maximumOpacity = max(maximumOpacity, max_opacities[hi]);
                lo = (lo+1)>>1;
                hi = (hi-1)>>1;

                int off = 0;
                size_t num_nodes = tf_width/2;
                while(lo <= hi)
                {
                    if ((lo & 1) == 1) maximumOpacity = max(maximumOpacity, max_opacities[off+lo]);
                    if ((hi & 1) == 0) maximumOpacity = max(maximumOpacity, max_opacities[off+hi]);
                    lo = (lo+1)>>1;
                    hi = (hi-1)>>1;
                    off += num_nodes;
                    num_nodes /= 2;
                }
            }
#else
            float maximumOpacity = max_opacities[ry * tf_width + rx];
#endif
#endif

            // Return the hit point if the grid cell is not fully transparent.
            if (maximumOpacity > 0.0f)
            {
                integrate(ray, t, t + delta, result.color);
                t += delta;
                continue;
            }

            // Exit bound of the grid cell in world coordinates.
            vec3f farBound(cellIndex + nextCellIndex << 4/*<< CELL_WIDTH_BITCOUNT*/);
            farBound /= vec3(vox-1);
            farBound *= bbox.size();
            farBound -= vec3(bbox.size().x/2, bbox.size().y/2, bbox.size().z/2);

            // Identify the distance along the ray to the exit points on the cell.
            const vec3f maximum = ray_rdir * (farBound - ray.ori);
            const float exitDist = min(min(tmax, maximum.x),
                                       min(maximum.y, maximum.z));

            const float dist = ceil(abs(exitDist - t) / delta) * delta;

            // Advance the ray so the next hit point will be outside the empty cell.
            t += dist;
        }

        return result;
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> ray_marching_traverse_leaves(R ray) const
    {
        using S = typename R::scalar_type;
        using C = vector<4, S>;

        result_record<S> result;
        result.color = C(0.0);

        vec3 inv_dir = 1.0f / ray.dir;

        for (int i = 0; i < num_leaves; ++i)
        {
            auto hit_rec = intersect(ray, leaves[i], inv_dir);
            result.hit |= hit_rec.hit;

            if (!hit_rec.hit)
                continue;

            auto t = max(S(0.0), hit_rec.tnear);
            auto tmax = hit_rec.tfar;
            integrate(ray, t, tmax, result.color);
        }

        return result;
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> ray_marching_traverse_hybrid(R ray) const
    {
        using S = typename R::scalar_type;
        using C = vector<4, S>;

        result_record<S> result;
        result.color = C(0.0);
        //result.color = C(temperature_to_rgb(0.f), 1.f);

        vec3 inv_dir = 1.0f / ray.dir;

        float const* max_opacities = max_opacities_;

        int num_steps = 0;
        for (int i = 0; i < num_leaves; ++i)
        {
            auto hit_rec = intersect(ray, leaves[i], inv_dir);
            result.hit |= hit_rec.hit;

            if (!hit_rec.hit)
                continue;

            auto t = max(S(0.0f), hit_rec.tnear);
            auto tmax = hit_rec.tfar;

            // cf. GridAccelerator_stepRay

            // Tentatively advance the ray.
            t += delta;

            const vec3f ray_rdir = 1.0f / ray.dir;
            // sign of direction determines near/far index
            const vec3i nextCellIndex = vec3i(1 - (*reinterpret_cast<unsigned*>(&ray.dir.x) >> 31),
                                              1 - (*reinterpret_cast<unsigned*>(&ray.dir.y) >> 31),
                                              1 - (*reinterpret_cast<unsigned*>(&ray.dir.z) >> 31));

            vec3i hit_cell;
            while (t < tmax)
            {
                // Compute the hit point in the local coordinate system.
                vec3f pos = ray.ori + t * ray.dir;
                vector<3, S> coord01(
                        (pos.x + (bbox.size().x / 2)) / bbox.size().x,
                        (pos.y + (bbox.size().y / 2)) / bbox.size().y,
                        (pos.z + (bbox.size().z / 2)) / bbox.size().z
                        );
                vec3f localCoordinates = coord01 * vec3(vox-1);

                // Compute the 3D index of the cell containing the hit point.
                vec3i cellIndex = vec3i(localCoordinates) >> 4;//>> CELL_WIDTH_BITCOUNT;

                // If we visited this cell before then it must not be empty.
                if (cellIndex == hit_cell)
                {
                    num_steps += 1;
                    integrate(ray, t, t + delta, result.color);
                    t += delta;
                    continue;
                }

                // Track the hit cell.
                hit_cell = cellIndex;

#if SIMPLE_GRID

                uint8_t empty = cells_empty[(grid_dims.z-cellIndex.z-1) * grid_dims.x * grid_dims.y + (grid_dims.y-cellIndex.y-1) * grid_dims.x + cellIndex.x];
                float maximumOpacity = empty ? .0f : 1.f;
#else
                // Get the volumetric value range of the cell.
                vec2f cellRange = cell_ranges[(grid_dims.z-cellIndex.z-1) * grid_dims.x * grid_dims.y + (grid_dims.y-cellIndex.y-1) * grid_dims.x + cellIndex.x];

                // Get the maximum opacity in the volumetric value range.
                int tf_width = TF_WIDTH;
                int rx = floor(cellRange.x * 255.0f);
                int ry = ceil(cellRange.y * 255.0f);
#ifdef RANGE_TREE
                float maximumOpacity = 0;
                {
                    int lo = rx;
                    int hi = ry-1;
                    if ((lo & 1) == 1) maximumOpacity = max(maximumOpacity, max_opacities[lo]);
                    if ((hi & 1) == 0) maximumOpacity = max(maximumOpacity, max_opacities[hi]);
                    lo = (lo+1)>>1;
                    hi = (hi-1)>>1;

                    int off = 0;
                    size_t num_nodes = tf_width/2;
                    while(lo <= hi)
                    {
                        if ((lo & 1) == 1) maximumOpacity = max(maximumOpacity, max_opacities[off+lo]);
                        if ((hi & 1) == 0) maximumOpacity = max(maximumOpacity, max_opacities[off+hi]);
                        lo = (lo+1)>>1;
                        hi = (hi-1)>>1;
                        off += num_nodes;
                        num_nodes /= 2;
                    }
                }
#else
                float maximumOpacity = max_opacities[ry * tf_width + rx];
#endif
#endif

                // Return the hit point if the grid cell is not fully transparent.
                if (maximumOpacity > 0.0f)
                {
                    num_steps += 1;
                    integrate(ray, t, t + delta, result.color);
                    t += delta;
                    continue;
                }

                // Exit bound of the grid cell in world coordinates.
                vec3f farBound(cellIndex + nextCellIndex << 4/*<< CELL_WIDTH_BITCOUNT*/);
                farBound /= vec3(vox-1);
                farBound *= bbox.size();
                farBound -= vec3(bbox.size().x/2, bbox.size().y/2, bbox.size().z/2);

                // Identify the distance along the ray to the exit points on the cell.
                const vec3f maximum = ray_rdir * (farBound - ray.ori);
                const float exitDist = min(min(tmax, maximum.x),
                                           min(maximum.y, maximum.z));

                const float dist = ceil(abs(exitDist - t) / delta) * delta;

                // Advance the ray so the next hit point will be outside the empty cell.
                t += dist;
            }
        }

        //result.color = C(temperature_to_rgb(num_steps / 512.f), 1.f);
        return result;
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> ray_marching_traverse_full(R ray) const
    {
        using S = typename R::scalar_type;
        using C = vector<4, S>;

        result_record<S> result;
        result.color = C(0.0);
        //result.color = C(temperature_to_rgb(0.f), 1.f);

        // traverse tree
        detail::stack<32> st;
        st.push(0);

        auto inv_dir = 1.0f / ray.dir;

        float t = 0.0f;

        auto get_bounds = [](virvo::SkipTreeNode& n)
        {
            return aabb(vec3(n.min_corner), vec3(n.max_corner));
        };

        auto get_child = [](virvo::SkipTreeNode& n, int index)
        {
            return index == 0 ? n.left : n.right;
        };

next:
        int num_steps = 0; // number of integration steps
        int num_boxes = 0;
        while (!st.empty())
        {
            auto node = nodes[st.pop()];

            while (node.left != -1 && node.right != -1)
            {
#if 1 // left and right are stored next to each other in memory
                auto children = &nodes[node.left];
#else // e.g. LBVH
                virvo::SkipTreeNode children[2] = { nodes[node.left], nodes[node.right] };
#endif

                auto hr1 = intersect(ray, get_bounds(children[0]), inv_dir);
                auto hr2 = intersect(ray, get_bounds(children[1]), inv_dir);
                num_boxes += 2;

                bool b1 = hr1.hit && hr1.tfar > t;
                bool b2 = hr2.hit && hr2.tfar > t;

                if (b1 && b2)
                {
                    unsigned near_addr = hr1.tnear < hr2.tnear ? 0 : 1;
                    st.push(get_child(node, !near_addr));
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
            auto hr = intersect(ray, get_bounds(node), inv_dir);
            //num_steps += (hr.tfar - hr.tnear) / delta;
            integrate(ray, hr.tnear, hr.tfar, result.color);
            t = max(t, hr.tfar - delta);
        }

        //result.color = C(temperature_to_rgb(num_boxes / 120.f), 1.f);
        //result.color = C(temperature_to_rgb(num_steps / 512.f), 1.f);

        return result;
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> operator()(R ray) const
    {
      switch (mode)
      {
      case Grid:
        return ray_marching_traverse_grid(ray);

      case Leaves:
        return ray_marching_traverse_leaves(ray);

      case Full:
        return ray_marching_traverse_full(ray);

      case Hybrid:
        return ray_marching_traverse_hybrid(ray);

      default:
      case Naive:
        return ray_marching_naive(ray);
      }
    }

    cuda_texture<unorm<8>, 3>::ref_type volume;
    cuda_texture<vec4, 1>::ref_type transfunc;

    aabb bbox;
    vec3i vox;
    float delta;
    bool local_shading;
    point_light<float> light;

    aabb const* leaves;
    int num_leaves;

    virvo::SkipTreeNode* nodes = nullptr;

    // Grid stuff
    vec2 const* cell_ranges;
    vec3i grid_dims;
    float const* max_opacities_; // TF_WIDTH * TF_WIDTH

    // This is the data for the _simple_ grid!
    uint8_t const* cells_empty;

    enum TraversalMode
    {
      Grid, Leaves, Full, Naive, Hybrid
    };

    TraversalMode mode;
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
    Impl()
        : sched(8, 8)
//      , tree(virvo::SkipTree::Grid)
//      , tree(virvo::SkipTree::LBVH)
        , tree(virvo::SkipTree::SVTKdTree)
//      , tree(virvo::SkipTree::SVTKdTreeCU)
        , grid(virvo::SkipTree::Grid)
    {
    }

    using R = basic_ray<float>;

    cuda_sched<R> sched;

    std::vector<cuda_texture<unorm<8>, 3>> volumes;

    cuda_texture<vec4, 1> transfunc;

    virvo::SkipTree tree;
    virvo::SkipTree grid; // hybrid grids: also have a grid

    virvo::SkipTreeNode* device_tree = nullptr;

    thrust::device_vector<aabb> d_leaves;

    // Ospray Grid Accelerator
    thrust::device_vector<virvo::vec2> d_cell_ranges;
    thrust::device_vector<float> d_max_opacities;

    // Simple grid (preclassified)
    thrust::device_vector<uint8_t> d_cells_empty;
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

    bool full = impl_->tree.getTechnique() == virvo::SkipTree::LBVH
             || impl_->tree.getTechnique() == virvo::SkipTree::SVTKdTree
             || impl_->tree.getTechnique() == virvo::SkipTree::SVTKdTreeCU;
    bool leaves = false;
    bool grid = impl_->tree.getTechnique() == virvo::SkipTree::Grid;
    bool hybrid = false;
    // Naive mode:
    if (0)
    {
        full = false;
        leaves = false;
        grid = false;
        hybrid = false;
    }
    // Hybrid mode:
    if (1)
    {
        full = false;
        leaves = false;
        grid = false;
        hybrid = true;
    }

    Kernel kernel;

    kernel.volume = cuda_texture<unorm<8>, 3>::ref_type(impl_->volumes[vd->getCurrentFrame()]);
    kernel.transfunc = cuda_texture<vec4, 1>::ref_type(impl_->transfunc);

    kernel.bbox          = aabb(vec3(bbox.min.data()), vec3(bbox.max.data()));
    kernel.vox           = vec3i(vd->vox[0], vd->vox[1], vd->vox[2]);
    kernel.delta         = delta;
    kernel.local_shading = getParameter(VV_LIGHTING);
    kernel.light         = light;

    if (full)
        kernel.nodes         = impl_->device_tree;

    std::vector<aabb> boxes;

    if (leaves || hybrid)
    {
        virvo::vec3 eye(getEyePosition().x, getEyePosition().y, getEyePosition().z);
        bool frontToBack = true;
        auto bricks = impl_->tree.getSortedBricks(eye, frontToBack);

        for (auto b : bricks)
        {
            boxes.push_back(aabb(vec3(b.min.data()), vec3(b.max.data())));
        }

        impl_->d_leaves = thrust::device_vector<aabb>(boxes);
    }

    virvo::CudaTimer t;
    if (full)
    {
        kernel.mode = Kernel::Full;
        auto sparams = make_sched_params(
            view_matrix,
            proj_matrix,
            virvo_rt
            );

        impl_->sched.frame(kernel, sparams);
    }
    else if (grid)
    {
        kernel.mode = Kernel::Grid;
        auto host_grid = impl_->tree.getMinMaxGrid();

        size_t len = host_grid.grid_dims.x * host_grid.grid_dims.y * host_grid.grid_dims.z;
        impl_->d_cell_ranges.resize(len);

        thrust::copy(host_grid.cell_ranges,
                     host_grid.cell_ranges + len,
                     impl_->d_cell_ranges.begin());
        kernel.cell_ranges = reinterpret_cast<vec2 const*>(
                    thrust::raw_pointer_cast(impl_->d_cell_ranges.data()));

        kernel.grid_dims = visionaray::vec3i(host_grid.grid_dims.data());

        int tf_width = TF_WIDTH;
        impl_->d_max_opacities.resize(tf_width*tf_width);

#ifdef RANGE_TREE
        thrust::copy(host_grid.max_opacities,
                     host_grid.max_opacities + tf_width-1,
                     impl_->d_max_opacities.begin());
        kernel.max_opacities_ = reinterpret_cast<float const*>(
                    thrust::raw_pointer_cast(impl_->d_max_opacities.data()));
#else
        thrust::copy(host_grid.max_opacities,
                     host_grid.max_opacities + tf_width*tf_width,
                     impl_->d_max_opacities.begin());
        kernel.max_opacities_ = reinterpret_cast<float const*>(
                    thrust::raw_pointer_cast(impl_->d_max_opacities.data()));
#endif

        auto sparams = make_sched_params(
            view_matrix,
            proj_matrix,
            virvo_rt
            );

        impl_->sched.frame(kernel, sparams);
    }
    else if (leaves)
    {
        kernel.mode = Kernel::Leaves;
        kernel.leaves = thrust::raw_pointer_cast(impl_->d_leaves.data());
        kernel.num_leaves = impl_->d_leaves.size();
        auto sparams = make_sched_params(
            view_matrix,
            proj_matrix,
            virvo_rt
            );

        impl_->sched.frame(kernel, sparams);
    }
    else if (hybrid)
    {
        kernel.mode = Kernel::Hybrid;
#if SIMPLE_GRID
        auto host_grid = impl_->tree.getSimpleGrid();

        kernel.cells_empty = thrust::raw_pointer_cast(impl_->d_cells_empty.data());
#else
        auto host_grid = impl_->grid.getMinMaxGrid();

        size_t len = host_grid.grid_dims.x * host_grid.grid_dims.y * host_grid.grid_dims.z;
        impl_->d_cell_ranges.resize(len);

        thrust::copy(host_grid.cell_ranges,
                     host_grid.cell_ranges + len,
                     impl_->d_cell_ranges.begin());
        kernel.cell_ranges = reinterpret_cast<vec2 const*>(
                    thrust::raw_pointer_cast(impl_->d_cell_ranges.data()));

        int tf_width = TF_WIDTH;
        impl_->d_max_opacities.resize(tf_width*tf_width);

#ifdef RANGE_TREE
        thrust::copy(host_grid.max_opacities,
                     host_grid.max_opacities + tf_width-1,
                     impl_->d_max_opacities.begin());
#else
        thrust::copy(host_grid.max_opacities,
                     host_grid.max_opacities + tf_width*tf_width,
                     impl_->d_max_opacities.begin());
#endif // RANGE_TREE
        kernel.max_opacities_ = reinterpret_cast<float const*>(
                    thrust::raw_pointer_cast(impl_->d_max_opacities.data()));

#endif
        kernel.grid_dims = visionaray::vec3i(host_grid.grid_dims.data());

        // Tree leaf nodes
        kernel.leaves = thrust::raw_pointer_cast(impl_->d_leaves.data());
        kernel.num_leaves = impl_->d_leaves.size();

        auto sparams = make_sched_params(
            view_matrix,
            proj_matrix,
            virvo_rt
            );

        impl_->sched.frame(kernel, sparams);
    }
    else
    {
        kernel.mode = Kernel::Naive;
        auto sparams = make_sched_params(
            view_matrix,
            proj_matrix,
            virvo_rt
            );

        impl_->sched.frame(kernel, sparams);
    }

    if (0)//_boundaries)
    {
        glEnable(GL_DEPTH_TEST);
        glDepthRange(0,0.95);
        glClearDepth(1.0f);
        glClear(GL_DEPTH_BUFFER_BIT);

        glLineWidth(3.0f);

        //glEnable(GL_LINE_SMOOTH);
        //glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

        //glEnable(GL_BLEND);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        bool isLightingEnabled = glIsEnabled(GL_LIGHTING);
        glDisable(GL_LIGHTING);

        virvo::vec4 clearColor = vvGLTools::queryClearColor();
        vvColor color(1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2]);
        impl_->tree.renderGL(color);

        //renderBoundingBox();

        if (isLightingEnabled)
            glEnable(GL_LIGHTING);
    }

    std::cout << std::fixed << std::setprecision(8);
    static double avg = 0.0;
    static size_t cnt = 0;
    avg += t.elapsed();
    cnt += 1;
    std::cout << avg/cnt << std::endl;
}

void vvSimpleCaster::updateTransferFunction()
{
    //for (int i = 0; i < 5; ++i)
    //for (int i = 0; i < 1100; ++i)
    {
    std::vector<vec4> tf(TF_WIDTH * 1 * 1);
    vd->computeTFTexture(0, TF_WIDTH, 1, 1, reinterpret_cast<float*>(tf.data()));

    impl_->transfunc = cuda_texture<vec4, 1>(tf.size());
    impl_->transfunc.reset(tf.data());
    impl_->transfunc.set_address_mode(Clamp);
    impl_->transfunc.set_filter_mode(Nearest);

    texture_ref<vec4, 1> tf_ref(tf.size());
    tf_ref.reset(tf.data());
    tf_ref.set_address_mode(Clamp);
    tf_ref.set_filter_mode(Nearest);

    impl_->tree.updateTransfunc(reinterpret_cast<const uint8_t*>(tf_ref.data()), TF_WIDTH, 1, 1, virvo::PF_RGBA32F);

    bool hybrid = true;
    if (hybrid)
    {
        impl_->grid.updateTransfunc(reinterpret_cast<const uint8_t*>(tf_ref.data()), TF_WIDTH, 1, 1, virvo::PF_RGBA32F);
    }

    auto simpleGrid = impl_->tree.getSimpleGrid();
    if (simpleGrid.cells != nullptr)
    {
      size_t len = simpleGrid.grid_dims.x * simpleGrid.grid_dims.y * simpleGrid.grid_dims.z;
      impl_->d_cells_empty.resize(len);
      thrust::copy(simpleGrid.cells,
                   simpleGrid.cells + len,
                   impl_->d_cells_empty.begin());
    }

    int numNodes = 0;
    impl_->device_tree = impl_->tree.getNodesDevPtr(numNodes);
//  std::cout << numNodes << '\n';
    }
    /*{
    impl_->tree.updateTransfunc(nullptr, 1, 1, 1, virvo::PF_RGBA32F);
    tex_filter_mode filter_mode = getParameter(VV_SLICEINT).asInt() == virvo::Linear ? Linear : Nearest;
    virvo::PixelFormat texture_format = virvo::PF_R8;
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
    }*/
}

void vvSimpleCaster::updateVolumeData()
{
    vvRenderer::updateVolumeData();

    impl_->tree.updateVolume(*vd);

    bool hybrid = true;
    if (hybrid)
        impl_->grid.updateVolume(*vd);


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

