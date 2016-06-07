// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <array>
#include <cassert>
#include <cstdlib>
#include <string>
#include <type_traits>

#include <GL/glew.h>

#ifdef VV_ARCH_CUDA
#include <thrust/device_vector.h>
#endif

#undef MATH_NAMESPACE

#include <visionaray/detail/pixel_access.h> // detail (TODO?)!
#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/bvh.h>
#include <visionaray/material.h>
#include <visionaray/packet_traits.h>
#include <visionaray/pixel_format.h>
#include <visionaray/pixel_traits.h>
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/scheduler.h>
#include <visionaray/shade_record.h>
#include <visionaray/traverse.h>

#ifdef VV_ARCH_CUDA
#include <visionaray/cuda/pixel_pack_buffer.h>
#endif

#undef MATH_NAMESPACE

#include "gl/util.h"
#include "vvcudarendertarget.h"
#include "vvraycaster.h"
#include "vvtoolshed.h"
#include "vvvoldesc.h"

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Ray type, depends upon target architecture
//

#if defined(VV_ARCH_SSE2) || defined(VV_ARCH_SSE4_1)
using ray_type = basic_ray<simd::float4>;
#elif defined(VV_ARCH_AVX) || defined(VV_ARCH_AVX2)
using ray_type = basic_ray<simd::float8>;
#else
using ray_type = basic_ray<float>;
#endif


//-------------------------------------------------------------------------------------------------
// Misc. helpers
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

template <typename F, typename I>
VSNRAY_FUNC
inline F normalize_depth(I const& depth, pixel_format depth_format, F /* */)
{
    if (depth_format == PF_DEPTH24_STENCIL8)
    {
        auto d = (depth & 0xFFFFFF00) >> 8;
        return F(d) / 16777215.0f;
    }

    // Assume PF_DEPTH32F
    return reinterpret_as_float(depth);
}

template <typename I1, typename I2, typename Params>
VSNRAY_FUNC
inline void get_depth(I1 x, I1 y, I2& depth_raw, Params const& params)
{
    if (params.depth_format == PF_DEPTH24_STENCIL8)
    {
        detail::pixel_access::get( // detail (TODO?)!
                pixel_format_constant<PF_DEPTH24_STENCIL8>{},
                x,
                y,
                params.viewport,
                depth_raw,
                params.depth_buffer
                );
    }
    else
    {
        // Assume PF_DEPTH32F
        detail::pixel_access::get( // detail (TODO?)!
                pixel_format_constant<PF_DEPTH32F>{},
                x,
                y,
                params.viewport,
                depth_raw,
                params.depth_buffer
                );
    }
}

VSNRAY_FUNC
inline vec3 gatherv(vec3 const* base_addr, int index)
{
    return base_addr[index];
}

template <
    typename T,
    typename I,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_CPU_FUNC
inline vector<3, T> gatherv(vector<3, T> const* base_addr, I const& index)
{
    // basically like visionaray::simd::gather, but
    // base_addr points to vec3's of simd-vectors

    typename simd::aligned_array<I>::type indices;
    store(indices, index);

    std::array<vector<3, float>, simd::num_elements<T>::value> arr;

    for (int i = 0; i < simd::num_elements<T>::value; ++i)
    {
        auto vecs = unpack(base_addr[indices[i]]);
        arr[i] = vecs[i];
    }

    return simd::pack(arr);
}


//-------------------------------------------------------------------------------------------------
// Clip sphere, hit_record stores both tnear and tfar (in contrast to basic_sphere)!
//

struct clip_sphere : basic_sphere<float>
{
};

template <typename T>
struct clip_sphere_hit_record
{
    using M = typename simd::mask_type<T>::type;

    M hit   = M(false);
    T tnear =  numeric_limits<T>::max();
    T tfar  = -numeric_limits<T>::max();
};

template <typename T>
VSNRAY_FUNC
inline clip_sphere_hit_record<T> intersect(basic_ray<T> const& ray, clip_sphere const& sphere)
{

    typedef basic_ray<T> ray_type;
    typedef vector<3, T> vec_type;

    ray_type r = ray;
    r.ori -= vec_type( sphere.center );

    auto A = dot(r.dir, r.dir);
    auto B = dot(r.dir, r.ori) * T(2.0);
    auto C = dot(r.ori, r.ori) - sphere.radius * sphere.radius;

    // solve Ax**2 + Bx + C
    auto disc = B * B - T(4.0) * A * C;
    auto valid = disc >= T(0.0);

    auto root_disc = select(valid, sqrt(disc), disc);

    auto q = select( B < T(0.0), T(-0.5) * (B - root_disc), T(-0.5) * (B + root_disc) );

    auto t1 = q / A;
    auto t2 = C / q;

    clip_sphere_hit_record<T> result;
    result.hit = valid;
    result.tnear   = select( valid, select( t1 > t2, t2, t1 ), T(-1.0) );
    result.tfar    = select( valid, select( t1 > t2, t1, t2 ), T(-1.0) );
    return result;
}


//-------------------------------------------------------------------------------------------------
// Clip box, basically an aabb, but intersect() returns a hit record containing the
// plane normal of the box' side where the ray entered
//

struct clip_box : basic_aabb<float>
{
    using base_type = basic_aabb<float>;

    VSNRAY_FUNC clip_box() = default;
    VSNRAY_FUNC clip_box(vec3 const& min, vec3 const& max)
        : base_type(min, max)
    {
    }
};

template <typename T>
struct clip_box_hit_record : hit_record<basic_ray<T>, basic_aabb<float>>
{
    vector<3, T> normal;
};

template <typename T>
VSNRAY_FUNC
inline clip_box_hit_record<T> intersect(basic_ray<T> const& ray, clip_box const& box)
{
    auto hr = intersect(ray, static_cast<clip_box::base_type>(box));

    // calculate normal
    vector<3, float> normals[6] {
            {  1.0f,  0.0f,  0.0f },
            { -1.0f,  0.0f,  0.0f },
            {  0.0f,  1.0f,  0.0f },
            {  0.0f, -1.0f,  0.0f },
            {  0.0f,  0.0f,  1.0f },
            {  0.0f,  0.0f, -1.0f }
            };

    auto isect_pos = ray.ori + ray.dir * hr.tnear;
    auto dir = normalize(isect_pos - vector<3, T>(box.center()));
    auto cosa = dot(dir, vector<3, T>(normals[0]));

    vector<3, T> normal(normals[0]);

    for (int i = 1; i < 6; ++i)
    {
        T dp    = dot(dir, vector<3, T>(normals[i]));
        normal  = select(dp > cosa, normals[i], normal);
        cosa    = select(dp > cosa, dp, cosa);
    }

    clip_box_hit_record<T> result;
    result.hit    = hr.hit;
    result.tnear  = hr.tnear;
    result.tfar   = hr.tfar;
    result.normal = normal;
    return result;
}


//-------------------------------------------------------------------------------------------------
// Clip plane (just another name for plane)
//

using clip_plane = basic_plane<3, float>;


//-------------------------------------------------------------------------------------------------
//
//

using clip_triangle_bvh = index_bvh_ref_t<basic_triangle<3, float>>;


//-------------------------------------------------------------------------------------------------
// Create clip intervals and deduce clip normals from primitive list
//

template <typename T>
struct clip_object_visitor
{
public:

    enum { MAX_INTERVALS = 64 };

    struct RT
    {
        int num_intervals;
        vector<2, T> intervals[MAX_INTERVALS];
        vector<3, T> normals[MAX_INTERVALS];
    };

    using return_type = RT;

public:

    // Create with ray and tnear / tfar obtained from ray / bbox intersection
    VSNRAY_FUNC
    clip_object_visitor(basic_ray<T> const& ray, T const& tnear, T const& tfar)
        : ray_(ray)
        , tnear_(tnear)
        , tfar_(tfar)
    {
    }

    // Clip plane
    VSNRAY_FUNC
    return_type operator()(clip_plane const& ref) const
    {
        auto hit_rec = intersect(ray_, ref);
        auto ndotd = dot(ray_.dir, vector<3, T>(ref.normal));

        return_type result;
        result.num_intervals  = 1;
        result.intervals[0].x = select(ndotd >  0.0f, hit_rec.t, tnear_);
        result.intervals[0].y = select(ndotd <= 0.0f, hit_rec.t, tfar_);
        result.normals[0]     = ref.normal;
        return result;
    }

    // Clip sphere
    VSNRAY_FUNC
    return_type operator()(clip_sphere const& ref) const
    {
        using V = vector<3, T>;

        auto hit_rec = intersect(ray_, ref);

        return_type result;
        result.num_intervals  = 1;
        result.intervals[0].x = select(hit_rec.tnear > tnear_, hit_rec.tnear, tnear_);
        result.intervals[0].y = select(hit_rec.tfar  < tfar_,  hit_rec.tfar,  tfar_);

        // normal at tfar, pointing inwards
        V isect_pos = ray_.ori + result.intervals[0].y * ray_.dir;
        result.normals[0]     = -(isect_pos - V(ref.center)) / T(ref.radius);

        return result;
    }

    // Clip triangles in a BVH
    VSNRAY_FUNC
    return_type operator()(clip_triangle_bvh const& ref) const
    {
        auto hit_rec = multi_hit<8>(ray_, &ref, &ref + 1);

        return_type result;
        T t = tnear_;

        result.num_intervals = 0;

        for (size_t i = 0; i < hit_rec.size(); i += 2)
        {
            auto hr1 = hit_rec[i];
            auto hr2 = hit_rec[i + 1];

            if (!any(hr1.hit) && !any(hr2.hit))
            {
                break;
            }

            auto n1 = get_normal(hit_rec[i], ref);
            auto n2 = get_normal(hit_rec[i + 1], ref);

            result.intervals[i / 2].x = select(
                    hr1.hit && hr2.hit,
                    hr1.t,
                    t
                    );
            result.intervals[i / 2].y = select(
                    hr1.hit && hr2.hit,
                    hr2.t,
                    hr1.t
                    );

            // Invalidate intervals where single rays
            // in a packet did not hit
            result.intervals[i / 2].x = select(
                    hr1.hit,
                    result.intervals[i / 2].x,
                    tfar_
                    );
            result.intervals[i / 2].y = select(
                    hr1.hit,
                    result.intervals[i / 2].y,
                    tnear_
                    );


            t = hr1.t;
            ++result.num_intervals;
        }
        return result;
    }

private:

    basic_ray<T>    ray_;
    T               tnear_;
    T               tfar_;
};


//-------------------------------------------------------------------------------------------------
// Wrapper that either uses CUDA/GL interop or simple CPU <- GPU transfer to make the
// OpenGL depth buffer available to the Visionaray kernel
//

#ifdef VV_ARCH_CUDA

struct depth_buffer_type : cuda::pixel_pack_buffer
{
    unsigned const* data() const
    {
        return static_cast<unsigned const*>(cuda::pixel_pack_buffer::data());
    }
};

#else

struct depth_buffer_type
{
    void map(recti viewport, pixel_format format)
    {
        auto info = map_pixel_format(format);

        buffer.resize((viewport.w - viewport.x) * (viewport.h - viewport.y));

        glReadPixels(
                viewport.x,
                viewport.y,
                viewport.w,
                viewport.h,
                info.format,
                info.type,
                buffer.data()
                );
    }

    void unmap()
    {
    }

    unsigned const* data() const
    {
        return buffer.data();
    }

    aligned_vector<unsigned> buffer;
};

#endif

//-------------------------------------------------------------------------------------------------
// Wrapper to consolidate virvo and Visionaray render targets
//

class virvo_render_target
{
public:

    static const pixel_format CF = PF_RGBA32F;
    static const pixel_format DF = PF_UNSPECIFIED;

    using color_type = typename pixel_traits<CF>::type;
    using depth_type = typename pixel_traits<DF>::type;

    using ref_type = render_target_ref<CF, DF>;

public:

    virvo_render_target(size_t w, size_t h, color_type* c, depth_type* d)
        : width_(w)
        , height_(h)
        , color_(c)
        , depth_(d)
    {
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }

    color_type* color() { return color_; }
    depth_type* depth() { return depth_; }

    color_type const* color() const { return color_; }
    depth_type const* depth() const { return depth_; }

    ref_type ref() { return ref_type(color(), depth()); }

    void begin_frame() {}
    void end_frame() {}

    size_t width_;
    size_t height_;

    color_type* color_;
    depth_type* depth_;
};


//-------------------------------------------------------------------------------------------------
// Visionaray volume rendering kernel
//

template <typename Params>
struct volume_kernel
{
    VSNRAY_FUNC
    explicit volume_kernel(Params const& p)
        : params(p)
    {
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> operator()(R ray, int x, int y) const
    {
        using S    = typename R::scalar_type;
        using I    = typename simd::int_type<S>::type;
        using Mask = typename simd::mask_type<S>::type;
        using Mat4 = matrix<4, 4, S>;
        using C    = vector<4, S>;

        result_record<S> result;
        result.color = C(0.0);

        auto hit_rec = intersect(ray, params.bbox);
        auto tmax = hit_rec.tfar;

        // convert depth buffer(x,y) to "t" coordinates
        if (params.depth_test)
        {
            // unproject (win to obj)
            I depth_raw(0);
            get_depth(x, y, depth_raw, params);
            S depth = normalize_depth(depth_raw, params.depth_format, S{});

            vector<3, S> win(expand_pixel<S>().x(x), expand_pixel<S>().y(y), depth);
            vector<4, S> u(
                    S(2.0 * (win[0] - params.viewport[0]) / params.viewport[2] - 1.0),
                    S(2.0 * (win[1] - params.viewport[1]) / params.viewport[3] - 1.0),
                    S(2.0 * win[2] - 1.0),
                    S(1.0)
                    );

            vector<4, S> v = Mat4(params.camera_matrix_inv) * u;
            vector<3, S> obj = v.xyz() / v.w;

            // convert to "t" coordinates
            tmax = length(obj - ray.ori);
        }


        auto t = max(S(0.0f), hit_rec.tnear);
        tmax = min(hit_rec.tfar, tmax);


        // calculate intervals clipped by planes, spheres, etc., along with the
        // normals of the farthest intersection in view direction
        const int MaxClipIntervals = 64;
        vector<2, S> clip_intervals[MaxClipIntervals];
        vector<3, S> clip_normals[MaxClipIntervals];

        int i = 0;
        for (auto it = params.clip_objects.begin; it != params.clip_objects.end; ++it)
        {
            clip_object_visitor<S> visitor(ray, t, tmax);
            auto clip_data = apply_visitor(visitor, *it);

            for (int j = 0; j < clip_data.num_intervals; ++j)
            {
                clip_intervals[i + j] = clip_data.intervals[j];
                clip_normals[i + j]   = clip_data.normals[j];
            }
            i += clip_data.num_intervals;

            if (i >= MaxClipIntervals)
            {
                // TODO: some error handling
                break;
            }
        }

        int num_clip_intervals = i;

        // treat the bbox entry plane as a clip
        // object that contributes a shading normal
        clip_normals[num_clip_intervals] = hit_rec.normal;


        // calculate the volume rendering integral
        while (visionaray::any(t < tmax))
        {
            Mask clipped(false);

            S tnext = t + params.delta;
            for (int i = 0; i < num_clip_intervals; ++i)
            {
                clipped |= t >= clip_intervals[i].x && t <= clip_intervals[i].y;
                tnext = select(
                        t >= clip_intervals[i].x && t <= clip_intervals[i].y && tnext < clip_intervals[i].y,
                        clip_intervals[i].y,
                        tnext
                        );
            }

            if (!visionaray::all(clipped))
            {
                auto pos = ray.ori + ray.dir * t;
                auto tex_coord = vector<3, S>(
                        ( pos.x + (params.bbox.size().x / 2) ) / params.bbox.size().x,
                        (-pos.y + (params.bbox.size().y / 2) ) / params.bbox.size().y,
                        (-pos.z + (params.bbox.size().z / 2) ) / params.bbox.size().z
                        );

                S voxel = tex3D(params.volume, tex_coord);
                C color = tex1D(params.transfunc, voxel);

                auto do_shade = params.local_shading && color.w >= 0.1f;

                if (visionaray::any(do_shade))
                {
                    // TODO: make this modifiable
                    plastic<S> mat;
                    mat.set_ca( from_rgb(vector<3, S>(0.3f, 0.3f, 0.3f)) );
                    mat.set_cd( from_rgb(vector<3, S>(0.8f, 0.8f, 0.8f)) );
                    mat.set_cs( from_rgb(vector<3, S>(0.8f, 0.8f, 0.8f)) );
                    mat.set_ka( 1.0f );
                    mat.set_kd( 1.0f );
                    mat.set_ks( 1.0f );
                    mat.set_specular_exp( 1000.0f );


                    // calculate shading
                    auto grad = gradient(params.volume, tex_coord);
                    auto normal = normalize(grad);

                    auto float_eq = [&](S const& a, S const& b) { return abs(a - b) < params.delta * S(0.5); };

                    Mask at_boundary = float_eq(t, hit_rec.tnear);
                    I clip_normal_index = select(
                            at_boundary,
                            I(num_clip_intervals), // bbox normal is stored at last position in the list
                            I(0)
                            );

                    for (int i = 0; i < num_clip_intervals; ++i)
                    {
                        Mask hit = float_eq(t, clip_intervals[i].y + params.delta); // TODO: understand why +delta
                        clip_normal_index = select(hit, I(i), clip_normal_index);
                        at_boundary |= hit;
                    }

                    if (visionaray::any(at_boundary))
                    {
                        auto boundary_normal = gatherv(clip_normals, clip_normal_index);
                        normal = select(
                                at_boundary,
                                boundary_normal * color.w + normal * (S(1.0) - color.w),
                                normal
                                );
                    }

                    do_shade &= length(grad) != 0.0f;

                    shade_record<point_light<float>, S> sr;
                    sr.isect_pos = pos;
                    sr.light = params.light;
                    sr.normal = normal;
                    sr.view_dir = -ray.dir;
                    sr.light_dir = normalize(sr.light.position());

                    auto shaded_clr = mat.shade(sr);

                    color.xyz() = mul(
                            color.xyz(),
                            to_rgb(shaded_clr),
                            do_shade,
                            color.xyz()
                            );
                }

                if (params.opacity_correction)
                {
                    color.w = 1.0f - pow(1.0f - color.w, params.delta);
                }

                // compositing
                if (params.mode == Params::AlphaCompositing)
                {
                    // premultiplied alpha
                    auto premult = color.xyz() * color.w;
                    color = C(premult, color.w);

                    result.color += select(
                            t < tmax && !clipped,
                            color * (1.0f - result.color.w),
                            C(0.0)
                            );

                    // early-ray termination - don't traverse w/o a contribution
                    if (params.early_ray_termination && visionaray::all(result.color.w >= 0.999f))
                    {
                        break;
                    }
                }
                else if (params.mode == Params::MaxIntensity)
                {
                    result.color = select(
                            t < tmax && !clipped,
                            max(color, result.color),
                            result.color
                            );
                }
                else if (params.mode == Params::MinIntensity)
                {
                    result.color = select(
                            t < tmax && !clipped,
                            min(color, result.color),
                            result.color
                            );
                }
                else if (params.mode == Params::DRR)
                {
                    result.color += select(
                            t < tmax && !clipped,
                            color,
                            C(0.0)
                            );
                }
            }

            // step on
            t = tnext;
        }

        result.hit = hit_rec.hit;
        return result;
    }

    Params params;
};


//-------------------------------------------------------------------------------------------------
// Volume kernel params
//

template <typename VolumeTex, typename TransfuncTex>
struct volume_kernel_params
{
    enum projection_mode
    {
        AlphaCompositing,
        MaxIntensity,
        MinIntensity,
        DRR
    };

    using volume_type    = VolumeTex;
    using transfunc_type = TransfuncTex;
    using clip_object    = variant<clip_plane, clip_sphere, clip_triangle_bvh>;

    clip_box            bbox;
    float               delta;
    VolumeTex           volume;
    TransfuncTex        transfunc;
    unsigned const*     depth_buffer;
    pixel_format        depth_format;
    projection_mode     mode;
    bool                depth_test;
    bool                opacity_correction;
    bool                early_ray_termination;
    bool                local_shading;
    mat4                camera_matrix_inv;
    recti               viewport;
    point_light<float>  light;

    struct
    {
        clip_object const* begin;
        clip_object const* end;
    } clip_objects;
};


//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct vvRayCaster::Impl
{
#if defined(VV_ARCH_CUDA)
    using sched_type        = cuda_sched<ray_type>;
    using volume8_type      = cuda_texture<unorm< 8>, NormalizedFloat, 3>;
    using volume16_type     = cuda_texture<unorm<16>, NormalizedFloat, 3>;
    using volume32_type     = cuda_texture<float,     ElementType,     3>;
    using transfunc_type    = cuda_texture<vec4,      ElementType,     1>;
#else
    using sched_type        = tiled_sched<ray_type>;
    using volume8_type      = texture<unorm< 8>, NormalizedFloat, 3>;
    using volume16_type     = texture<unorm<16>, NormalizedFloat, 3>;
    using volume32_type     = texture<float,     ElementType,     3>;
    using transfunc_type    = texture<vec4,      ElementType,     1>;

    Impl()
        : sched(vvToolshed::getNumProcessors())
    {
        char* num_threads = getenv("VV_NUM_THREADS");
        if (num_threads != nullptr)
        {
            std::string str(num_threads);
            sched.set_num_threads(std::stoi(str));
        }
    }
#endif

    using params8_type      = volume_kernel_params<typename volume8_type::ref_type,  typename transfunc_type::ref_type>;
    using params16_type     = volume_kernel_params<typename volume16_type::ref_type, typename transfunc_type::ref_type>;
    using params32_type     = volume_kernel_params<typename volume32_type::ref_type, typename transfunc_type::ref_type>;


    sched_type                      sched;
    params8_type                    params8;
    params16_type                   params16;
    params32_type                   params32;
    std::vector<volume8_type>       volume8;
    std::vector<volume16_type>      volume16;
    std::vector<volume32_type>      volume32;
    transfunc_type                  transfunc;
    depth_buffer_type               depth_buffer;

    void updateVolumeTextures(vvVolDesc* vd, vvRenderer* renderer);
    void updateTransfuncTexture(vvVolDesc* vd, vvRenderer* renderer);
};


void vvRayCaster::Impl::updateVolumeTextures(vvVolDesc* vd, vvRenderer* renderer)
{
    tex_filter_mode filter_mode = renderer->getParameter(vvRenderer::VV_SLICEINT).asInt() == virvo::Linear ? Linear : Nearest;
    tex_address_mode address_mode = Clamp;

    if (vd->bpc == 1)
    {
        volume8.resize(vd->frames);
        for (size_t f = 0; f < vd->frames; ++f)
        {
            volume8[f].resize(vd->vox[0], vd->vox[1], vd->vox[2]);
            volume8[f].set_data(reinterpret_cast<unorm<8> const*>(vd->getRaw(f)));
            volume8[f].set_address_mode(address_mode);
            volume8[f].set_filter_mode(filter_mode);
        }
    }
    else if (vd->bpc == 2)
    {
        volume16.resize(vd->frames);
        for (size_t f = 0; f < vd->frames; ++f)
        {
            volume16[f].resize(vd->vox[0], vd->vox[1], vd->vox[2]);
            volume16[f].set_data(reinterpret_cast<unorm<16> const*>(vd->getRaw(f)));
            volume16[f].set_address_mode(address_mode);
            volume16[f].set_filter_mode(filter_mode);
        }
    }
    else if (vd->bpc == 4)
    {
        volume32.resize(vd->frames);
        for (size_t f = 0; f < vd->frames; ++f)
        {
            volume32[f].resize(vd->vox[0], vd->vox[1], vd->vox[2]);
            volume32[f].set_data(reinterpret_cast<float const*>(vd->getRaw(f)));
            volume32[f].set_address_mode(address_mode);
            volume32[f].set_filter_mode(filter_mode);
        }
    }
}

void vvRayCaster::Impl::updateTransfuncTexture(vvVolDesc* vd, vvRenderer* /*renderer*/)
{
    aligned_vector<vec4> tf(256 * 1 * 1);
    vd->computeTFTexture(0, 256, 1, 1, reinterpret_cast<float*>(tf.data()));

    transfunc.resize(tf.size());
    transfunc.set_data(tf.data());
    transfunc.set_address_mode(Clamp);
    transfunc.set_filter_mode(Nearest);
}


//-------------------------------------------------------------------------------------------------
// Public interface
//

vvRayCaster::vvRayCaster(vvVolDesc* vd, vvRenderState renderState)
    : vvRenderer(vd, renderState)
    , impl_(new Impl)
{
    rendererType = RAYREND;

    glewInit();

#if defined(VV_ARCH_CUDA)
    virvo::RenderTarget* rt = virvo::PixelUnpackBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);

    // no direct rendering
    if (rt == NULL)
    {
        rt = virvo::DeviceBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);
    }
    setRenderTarget(rt);
#else
    setRenderTarget(virvo::HostBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED));
#endif

    updateVolumeData();
    updateTransferFunction();
}

vvRayCaster::~vvRayCaster()
{
}

void vvRayCaster::renderVolumeGL()
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
        viewport,
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

    // Get OpenGL depth buffer to clip against
    pixel_format depth_format = PF_UNSPECIFIED;

    bool depth_test = glIsEnabled(GL_DEPTH_TEST);

    if (depth_test)
    {
        GLint depth_bits = 0;
        glGetFramebufferAttachmentParameteriv(
                GL_FRAMEBUFFER,
                GL_DEPTH,
                GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,
                &depth_bits
                );

        GLint stencil_bits = 0;
        glGetFramebufferAttachmentParameteriv(
                GL_FRAMEBUFFER,
                GL_STENCIL,
                GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE,
                &stencil_bits
                );


        // TODO: make this more general
        // 24-bit depth buffer and 8-bit stencil buffer
        // is however a quite common case
        depth_format = (depth_bits == 24 && stencil_bits == 8) ? PF_DEPTH24_STENCIL8 : PF_DEPTH32F;

#ifdef __APPLE__
        // PIXEL_PACK_BUFFER with unsigned does not work
        // on Mac OS X, default to 32-bit floating point
        // depth buffer
        depth_format = PF_DEPTH32F;
#endif

        impl_->depth_buffer.map(viewport, depth_format);
        depth_test = true;
    }

    // assemble clip objects
    aligned_vector<typename Impl::params8_type::clip_object> clip_objects;

#if 0
    // OpenGL clip planes
    for (int i = 0; i < GL_MAX_CLIP_PLANES; ++i)
    {
        if (!glIsEnabled(GL_CLIP_PLANE0 + i))
        {
            continue;
        }

        GLdouble eq[4] = { 0, 0, 0, 0 };
        glGetClipPlane(GL_CLIP_PLANE0 + i, eq);

        clip_plane pl;
        pl.normal = vec3(eq[0], eq[1], eq[2]);
        pl.offset = eq[3];
        clip_objects.push_back(pl);
    }
#else

    // Uncomment to test clipping with sphere
#if 0
    auto sphere = vvClipSphere::create();
    sphere->center = virvo::vec3(0, 0, 50);
    sphere->radius = 50.0f;
    setParameter(VV_CLIP_OBJ6, sphere);
    setParameter(VV_CLIP_OBJ_ACTIVE6, true);
#endif

    // Uncomment to test clipping with a simple triangle mesh
#if 1
    using TT = vvClipTriangleList::Triangle;
    auto mesh = vvClipTriangleList::create();
    mesh->resize(4);
    (*mesh)[0] = TT{{ -1.0f,  0.0f,  1.0f }, {  1.0f,  0.0f,  1.0f }, {  0.0f,  1.0f,  0.0f }};
    (*mesh)[1] = TT{{  1.0f,  0.0f,  1.0f }, {  0.0f,  0.0f, -1.0f }, {  0.0f,  1.0f,  0.0f }};
    (*mesh)[2] = TT{{  0.0f,  0.0f, -1.0f }, { -1.0f,  0.0f,  1.0f }, {  0.0f,  1.0f,  0.0f }};
    (*mesh)[3] = TT{{  1.0f,  0.0f,  1.0f }, { -1.0f,  0.0f,  1.0f }, {  0.0f,  0.0f, -1.0f }};
    setParameter(VV_CLIP_OBJ7, mesh);
    setParameter(VV_CLIP_OBJ_ACTIVE7, true);
#endif


    using Triangle = basic_triangle<3, float>;
    index_bvh<Triangle> bvhs[NUM_CLIP_OBJS];

    typedef vvRenderState::ParameterType PT;
    PT act_id = VV_CLIP_OBJ_ACTIVE0;
    PT obj_id = VV_CLIP_OBJ0;

    for ( ; act_id != VV_CLIP_OBJ_ACTIVE_LAST && obj_id != VV_CLIP_OBJ_LAST
          ; act_id = PT(act_id + 1), obj_id = PT(obj_id + 1))
    {
        if (getParameter(act_id))
        {
            auto obj = getParameter(obj_id).asClipObj();

            if (auto plane = boost::dynamic_pointer_cast<vvClipPlane>(obj))
            {
                clip_plane pl;
                pl.normal = vec3(plane->normal.x, plane->normal.y, plane->normal.z);
                pl.offset = plane->offset;
                clip_objects.push_back(pl);
            }
            else if (auto sphere = boost::dynamic_pointer_cast<vvClipSphere>(obj))
            {
                clip_sphere sp;
                sp.center = vec3(sphere->center.x, sphere->center.y, sphere->center.z);
                sp.radius = sphere->radius;
                clip_objects.push_back(sp);
            }
            else if (auto mesh = boost::dynamic_pointer_cast<vvClipTriangleList>(obj))
            {
                using Triangle = basic_triangle<3, float>;

                auto bvh_id = VV_CLIP_OBJ_LAST - obj_id;

                aligned_vector<Triangle> tris;

                // Convert to Visionaray layout and assign IDs
                int prim_id = 0;
                for (auto t : *mesh)
                {
                    Triangle vt;
                    vt.v1 = vec3(t.v1.x, t.v1.y, t.v1.z) * 50.0f;
                    vt.e1 = vec3(t.v2.x, t.v2.y, t.v2.z) * 50.0f- vt.v1;
                    vt.e2 = vec3(t.v3.x, t.v3.y, t.v3.z) * 50.0f- vt.v1;
                    vt.prim_id = prim_id++;
                    vt.geom_id = 0;
                    tris.push_back(vt);
                }

                // Build up the BVH
                bvhs[bvh_id] = build<index_bvh<Triangle>>(tris.data(), tris.size());
                clip_objects.push_back(bvhs[bvh_id].ref());
            }
        }
    }
#endif


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


#ifdef VV_ARCH_CUDA
    // TODO: consolidate!
    thrust::device_vector<typename Impl::params8_type::clip_object> device_objects(clip_objects);
    auto clip_objects_begin = [&]()
    {
        return thrust::raw_pointer_cast(device_objects.data());
    };

    auto clip_objects_end = [&]()
    {
        return clip_objects_begin() + device_objects.size();
    };
#else
    auto clip_objects_begin = [&]()
    {
        return clip_objects.data();
    };

    auto clip_objects_end = [&]()
    {
        return clip_objects.data() + clip_objects.size();
    };
#endif


    // assemble volume kernel params and call kernel
    if (vd->bpc == 1)
    {
        impl_->params8.bbox                     = clip_box( vec3(bbox.min.data()), vec3(bbox.max.data()) );
        impl_->params8.delta                    = delta;
        impl_->params8.volume                   = Impl::params8_type::volume_type(impl_->volume8[vd->getCurrentFrame()]);
        impl_->params8.transfunc                = Impl::params8_type::transfunc_type(impl_->transfunc);
        impl_->params8.depth_buffer             = impl_->depth_buffer.data();
        impl_->params8.depth_format             = depth_format;
        impl_->params8.mode                     = Impl::params8_type::projection_mode(getParameter(VV_MIP_MODE).asInt());
        impl_->params8.depth_test               = depth_test;
        impl_->params8.opacity_correction       = getParameter(VV_OPCORR);
        impl_->params8.early_ray_termination    = getParameter(VV_TERMINATEEARLY);
        impl_->params8.local_shading            = getParameter(VV_LIGHTING);
        impl_->params8.camera_matrix_inv        = inverse(proj_matrix * view_matrix);
        impl_->params8.viewport                 = viewport;
        impl_->params8.light                    = light;
        impl_->params8.clip_objects.begin       = clip_objects_begin();
        impl_->params8.clip_objects.end         = clip_objects_end();

        volume_kernel<Impl::params8_type> kernel(impl_->params8);
        impl_->sched.frame(kernel, sparams);
    }
    else if (vd->bpc == 2)
    {
        impl_->params16.bbox                    = clip_box( vec3(bbox.min.data()), vec3(bbox.max.data()) );
        impl_->params16.delta                   = delta;
        impl_->params16.volume                  = Impl::params16_type::volume_type(impl_->volume16[vd->getCurrentFrame()]);
        impl_->params16.transfunc               = Impl::params16_type::transfunc_type(impl_->transfunc);
        impl_->params16.depth_buffer            = impl_->depth_buffer.data();
        impl_->params16.depth_format            = depth_format;
        impl_->params16.mode                    = Impl::params16_type::projection_mode(getParameter(VV_MIP_MODE).asInt());
        impl_->params16.depth_test              = depth_test;
        impl_->params16.opacity_correction      = getParameter(VV_OPCORR);
        impl_->params16.early_ray_termination   = getParameter(VV_TERMINATEEARLY);
        impl_->params16.local_shading           = getParameter(VV_LIGHTING);
        impl_->params16.camera_matrix_inv       = inverse(proj_matrix * view_matrix);
        impl_->params16.viewport                = viewport;
        impl_->params16.light                   = light;
        impl_->params16.clip_objects.begin      = clip_objects_begin();
        impl_->params16.clip_objects.end        = clip_objects_end();

        volume_kernel<Impl::params16_type> kernel(impl_->params16);
        impl_->sched.frame(kernel, sparams);
    }
    else if (vd->bpc == 4)
    {
        impl_->params32.bbox                    = clip_box( vec3(bbox.min.data()), vec3(bbox.max.data()) );
        impl_->params32.delta                   = delta;
        impl_->params32.volume                  = Impl::params32_type::volume_type(impl_->volume32[vd->getCurrentFrame()]);
        impl_->params32.transfunc               = Impl::params32_type::transfunc_type(impl_->transfunc);
        impl_->params32.depth_buffer            = impl_->depth_buffer.data();
        impl_->params32.depth_format            = depth_format;
        impl_->params32.mode                    = Impl::params32_type::projection_mode(getParameter(VV_MIP_MODE).asInt());
        impl_->params32.depth_test              = depth_test;
        impl_->params32.opacity_correction      = getParameter(VV_OPCORR);
        impl_->params32.early_ray_termination   = getParameter(VV_TERMINATEEARLY);
        impl_->params32.local_shading           = getParameter(VV_LIGHTING);
        impl_->params32.camera_matrix_inv       = inverse(proj_matrix * view_matrix);
        impl_->params32.viewport                = viewport;
        impl_->params32.light                   = light;
        impl_->params32.clip_objects.begin      = clip_objects_begin();
        impl_->params32.clip_objects.end        = clip_objects_end();

        volume_kernel<Impl::params32_type> kernel(impl_->params32);
        impl_->sched.frame(kernel, sparams);
    }

    if (depth_test)
    {
        impl_->depth_buffer.unmap();
    }
}

void vvRayCaster::updateTransferFunction()
{
    impl_->updateTransfuncTexture(vd, this);
}

void vvRayCaster::updateVolumeData()
{
    impl_->updateVolumeTextures(vd, this);
}

bool vvRayCaster::checkParameter(ParameterType param, vvParam const& value) const
{
    switch (param)
    {
    case VV_SLICEINT:
        {
            virvo::tex_filter_mode mode = static_cast< virvo::tex_filter_mode >(value.asInt());

            if (mode == virvo::Nearest || mode == virvo::Linear)
            {
                return true;
            }
        }
        return false;

    case VV_CLIP_OBJ0:
    case VV_CLIP_OBJ1:
    case VV_CLIP_OBJ2:
    case VV_CLIP_OBJ3:
    case VV_CLIP_OBJ4:
    case VV_CLIP_OBJ5:
    case VV_CLIP_OBJ6:
    case VV_CLIP_OBJ7:
        return true;

    default:
        return vvRenderer::checkParameter(param, value);
    }
}

void vvRayCaster::setParameter(ParameterType param, vvParam const& value)
{
    switch (param)
    {
    case VV_SLICEINT:
        {
            if (_interpolation != static_cast< virvo::tex_filter_mode >(value.asInt()))
            {
                _interpolation = static_cast< virvo::tex_filter_mode >(value.asInt());
                tex_filter_mode filter_mode = _interpolation == virvo::Linear ? Linear : Nearest;

                for (auto& tex : impl_->volume8)
                {
                    tex.set_filter_mode(filter_mode);
                }

                for (auto& tex : impl_->volume16)
                {
                    tex.set_filter_mode(filter_mode);
                }

                for (auto& tex : impl_->volume32)
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

bool vvRayCaster::instantClassification() const
{
    return true;
}

vvRenderer* createRayCaster(vvVolDesc* vd, vvRenderState const& rs)
{
    return new vvRayCaster(vd, rs);
}
