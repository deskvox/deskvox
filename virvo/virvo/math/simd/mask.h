#pragma once

#include "vec.h"
#include "veci.h"
#include "../vector.h"


namespace MATH_NAMESPACE
{


namespace simd
{


union mask4
{
public:

    __m128  f;
    __m128i i;

    mask4()
    {
    }

    VV_FORCE_INLINE mask4(__m128 m)
        : f(m)
    {
    }

    VV_FORCE_INLINE mask4(__m128i m)
        : i(m)
    {
    }

    VV_FORCE_INLINE mask4(float4 const& m)
        : f(m)
    {
    }

    VV_FORCE_INLINE operator float4() const
    {
        return f;
    }

};


/* float4 */

VV_FORCE_INLINE mask4 operator<(float4 const& u, float4 const& v)
{
  return _mm_cmplt_ps(u, v);
}

VV_FORCE_INLINE mask4 operator>(float4 const& u, float4 const& v)
{
  return _mm_cmpgt_ps(u, v);
}

VV_FORCE_INLINE mask4 operator<=(float4 const& u, float4 const& v)
{
  return _mm_cmple_ps(u, v);
}

VV_FORCE_INLINE mask4 operator>=(float4 const& u, float4 const& v)
{
  return _mm_cmpge_ps(u, v);
}

VV_FORCE_INLINE mask4 operator==(float4 const& u, float4 const& v)
{
  return _mm_cmpeq_ps(u, v);
}

VV_FORCE_INLINE mask4 operator!=(float4 const& u, float4 const& v)
{
  return _mm_cmpneq_ps(u, v);
}

VV_FORCE_INLINE mask4 operator&&(float4 const& u, float4 const& v)
{
  return _mm_and_ps(u, v);
}

VV_FORCE_INLINE mask4 operator||(float4 const& u, float4 const& v)
{
  return _mm_and_ps(u, v);
}


VV_FORCE_INLINE bool any(mask4 const& m)
{
    return _mm_movemask_ps(m.f) != 0;
}

VV_FORCE_INLINE bool all(mask4 const& m)
{
    return _mm_movemask_ps(m.f) == 0xF;
}

VV_FORCE_INLINE float4 select(mask4 const& m, float4 const& a, float4 const& b)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
    return _mm_blendv_ps(b, a, m.f);
#else
    return _mm_or_ps(_mm_and_ps(m.f, a), _mm_andnot_ps(m.f, b));
#endif
}

VV_FORCE_INLINE int4 select(mask4 const& m, int4 const& a, int4 const& b)
{
    return reinterpret_as_int( select(m, reinterpret_as_float(a), reinterpret_as_float(b)) );
}

VV_FORCE_INLINE mask4 operator&(mask4 const& a, mask4 const& b)
{
    return _mm_and_si128(a.i, b.i);
}

VV_FORCE_INLINE mask4 operator|(mask4 const &a, mask4 const& b)
{
    return _mm_or_si128(a.i, b.i);
}

VV_FORCE_INLINE mask4 operator^(mask4 const& a, mask4 const& b)
{
    return _mm_xor_si128(a.i, b.i);
}

VV_FORCE_INLINE mask4& operator&=(mask4& a, mask4 const& b)
{
    a = a & b;
    return a;
}

VV_FORCE_INLINE mask4& operator|=(mask4& a, mask4 const& b)
{
    a = a | b;
    return a;
}

VV_FORCE_INLINE mask4& operator^=(mask4& a, mask4 const& b)
{
    a = a ^ b;
    return a;
}

VV_FORCE_INLINE float4 neg(float4 const& v, mask4 const& mask)
{
    return select(mask, -v, 0.0f);
}

VV_FORCE_INLINE float4 add(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, u + v, 0.0f);
}

VV_FORCE_INLINE float sub(float u, float v, float /* mask */)
{
  return u - v;
}

VV_FORCE_INLINE float4 sub(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, u - v, 0.0f);
}

VV_FORCE_INLINE float4 mul(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, u * v, 0.0f);
}

VV_FORCE_INLINE float4 div(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, u / v, 0.0f);
}

VV_FORCE_INLINE float4 lt(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u < v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 gt(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u > v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 le(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u <= v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 ge(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u >= v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 eq(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u == v).f, float4(0.0f));
}

VV_FORCE_INLINE float4 neq(float4 const& u, float4 const& v, mask4 const& mask)
{
    return select(mask, mask4(u != v).f, float4(0.0f));
}

VV_FORCE_INLINE void store(float dst[4], float4 const& v, mask4 const& mask)
{
    float4 tmp = select(mask, v, float4(0.0f));
    store(dst, tmp);
}


/* Vec3 */

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > neg(vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( neg(v.x, mask), neg(v.y, mask), neg(v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > add(vector< 3, T > const& u, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( add(u.x, v.x, mask), add(u.y, v.y, mask), add(u.z, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > sub(vector< 3, T > const& u, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( sub(u.x, v.x, mask), sub(u.y, v.y, mask), sub(u.z, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > mul(vector< 3, T > const& u, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( mul(u.x, v.x, mask), mul(u.y, v.y, mask), mul(u.z, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > div(vector< 3, T > const& u, vector< 3, T > const& v, M const& mask)
{
    return vector<3, T >( div(u.x, v.x, mask), div(u.y, v.y, mask), div(u.z, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > add(vector< 3, T > const& v, T const& s, M const& mask)
{
    return vector< 3, T >( add(v.x, s, mask), add(v.y, s, mask), add(v.z, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > sub(vector< 3, T > const& v, T const& s, M const& mask)
{
    return vector< 3, T >( sub(v.x, s, mask), sub(v.y, s, mask), sub(v.z, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > mul(vector< 3, T > const& v, T const& s, M const& mask)
{
    return vector< 3, T >( mul(v.x, s, mask), mul(v.y, s, mask), mul(v.z, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > div(vector< 3, T > const& v, T const& s, M const& mask)
{
    return vector< 3, T >( div(v.x, s, mask), div(v.y, s, mask), div(v.z, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > add(T const& s, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( add(s, v.x, mask), add(s, v.y, mask), add(s, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > sub(T const& s, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( sub(s, v.x, mask), sub(s, v.y, mask), sub(s, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > mul(T const& s, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( mul(s, v.x, mask), mul(s, v.y, mask), sub(s, v.z, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 3, T > div(T const& s, vector< 3, T > const& v, M const& mask)
{
    return vector< 3, T >( div(s, v.x, mask), div(s, v.y, mask), div(s, v.z, mask) );
}


/* Vec4 */

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > neg(vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( neg(v.x, mask), neg(v.y, mask), neg(v.z, mask), neg(v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > add(vector< 4, T > const& u, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( add(u.x, v.x, mask), add(u.y, v.y, mask), add(u.z, v.z, mask), add(u.w, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > sub(vector< 4, T > const& u, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( sub(u.x, v.x, mask), sub(u.y, v.y, mask), sub(u.z, v.z, mask), sub(u.w, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > mul(vector< 4, T > const& u, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( mul(u.x, v.x, mask), mul(u.y, v.y, mask), mul(u.z, v.z, mask), mul(u.w, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > div(vector< 4, T > const& u, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( div(u.x, v.x, mask), div(u.y, v.y, mask), div(u.z, v.z, mask), div(u.w, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > add(vector< 4, T > const& v, T const& s, M const& mask)
{
    return vector< 4, T >( add(v.x, s, mask), add(v.y, s, mask), add(v.z, s, mask), add(v.w, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > sub(vector< 4, T > const& v, T const& s, M const& mask)
{
    return vector< 4, T >( sub(v.x, s, mask), sub(v.y, s, mask), sub(v.z, s, mask), sub(v.w, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > mul(vector< 4, T > const& v, T const& s, M const& mask)
{
    return vector< 4, T >( mul(v.x, s, mask), mul(v.y, s, mask), mul(v.z, s, mask), mul(v.w, s, mask) );
}

VV_FORCE_INLINE vector< 4, float > mul(vector< 4, float > const& v, float s, float /* mask */)
{
    return v * s;
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > div(vector< 4, T > const& v, T const& s, M const& mask)
{
    return vector< 4, T >( div(v.x, s, mask), div(v.y, s, mask), div(v.z, s, mask), div(v.w, s, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > add(T const& s, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( add(s, v.x, mask), add(s, v.y, mask), add(s, v.z, mask), add(s, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > sub(T const& s, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( sub(s, v.x, mask), sub(s, v.y, mask), sub(s, v.z, mask), sub(s, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > mul(T const& s, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( mul(s, v.x, mask), mul(s, v.y, mask), mul(s, v.z, mask), mul(s, v.w, mask) );
}

template < typename T, typename M >
VV_FORCE_INLINE vector< 4, T > div(T const& s, vector< 4, T > const& v, M const& mask)
{
    return vector< 4, T >( div(s, v.x, mask), div(s, v.y, mask), div(s, v.z, mask), div(s, v.w, mask) );
}


} // simd


} // MATH_NAMESPACE


