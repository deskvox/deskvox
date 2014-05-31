#pragma once

#include "vec.h"
#include "../vector.h"

#include <ostream>

namespace virvo
{
namespace math
{


template < >
class Matrix< simd::float4 >
{
public:

  typedef simd::float4 row_type;

  inline Matrix< simd::float4 >()
  {
  }

    inline Matrix(float* data)
    {

        rows[0] = row_type( &data[ 0] );
        rows[1] = row_type( &data[ 4] );
        rows[2] = row_type( &data[ 8] );
        rows[3] = row_type( &data[12] );

    }

/*  inline Matrix< simd::float4 >(virvo::Matrix const& m)
  {
    VV_ALIGN(16) float row1[4];
    VV_ALIGN(16) float row2[4];
    VV_ALIGN(16) float row3[4];
    VV_ALIGN(16) float row4[4];

    m.getRow(0, &row1[0], &row1[1], &row1[2], &row1[3]);
    m.getRow(1, &row2[0], &row2[1], &row2[2], &row2[3]);
    m.getRow(2, &row3[0], &row3[1], &row3[2], &row3[3]);
    m.getRow(3, &row4[0], &row4[1], &row4[2], &row4[3]);

    rows[0] = row1;
    rows[1] = row2;
    rows[2] = row3;
    rows[3] = row4;
  }

  inline operator virvo::Matrix() const
  {
    VV_ALIGN(16) float row1[4];
    VV_ALIGN(16) float row2[4];
    VV_ALIGN(16) float row3[4];
    VV_ALIGN(16) float row4[4];

    store(rows[0], &row1[0]);
    store(rows[1], &row2[0]);
    store(rows[2], &row3[0]);
    store(rows[3], &row4[0]);

    virvo::Matrix m;
    m.setRow(0, row1[0], row1[1], row1[2], row1[3]);
    m.setRow(1, row2[0], row2[1], row2[2], row2[3]);
    m.setRow(2, row3[0], row3[1], row3[2], row3[3]);
    m.setRow(3, row4[0], row4[1], row4[2], row4[3]);
    return m;
  } */

  inline simd::float4 row(size_t i) const
  {
    assert(i < 4);
    return rows[i];
  }

  inline void setRow(size_t i, simd::float4 const& v)
  {
    assert(i < 4);
    rows[i] = v;
  }

  void identity()
  {
    using simd::float4;

    rows[0] = float4(1.0f, 0.0f, 0.0f, 0.0f);
    rows[1] = float4(0.0f, 1.0f, 0.0f, 0.0f);
    rows[2] = float4(0.0f, 0.0f, 1.0f, 0.0f);
    rows[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);
  }

  void transpose()
  {
    using simd::float4;

    float4 tmp1 = _mm_unpacklo_ps(rows[0], rows[1]);
    float4 tmp2 = _mm_unpacklo_ps(rows[2], rows[3]);
    float4 tmp3 = _mm_unpackhi_ps(rows[0], rows[1]);
    float4 tmp4 = _mm_unpackhi_ps(rows[2], rows[3]);

    rows[0] = _mm_movelh_ps(tmp1, tmp2);
    rows[1] = _mm_movehl_ps(tmp2, tmp1);
    rows[2] = _mm_movelh_ps(tmp3, tmp4);
    rows[3] = _mm_movehl_ps(tmp4, tmp3);
  }

  void invert()
  {
    using simd::float4;
    using simd::shuffle;

    float4 cofactors[6];

    float4 tmpa = shuffle<3, 3, 3, 3>(rows[3], rows[2]);
    float4 tmpb = shuffle<2, 2, 2, 2>(rows[3], rows[2]);
    float4 tmp0 = shuffle<2, 2, 2, 2>(rows[2], rows[1]);
    float4 tmp1 = shuffle<0, 0, 0, 2>(tmpa, tmpa);
    float4 tmp2 = shuffle<0, 0, 0, 2>(tmpb, tmpb);
    float4 tmp3 = shuffle<3, 3, 3, 3>(rows[2], rows[1]);
    cofactors[0] = tmp0 * tmp1 - tmp2 * tmp3;

    tmpa = shuffle<3, 3, 3, 3>(rows[3], rows[2]);
    tmpb = shuffle<1, 1, 1, 1>(rows[3], rows[2]);
    tmp0 = shuffle<1, 1, 1, 1>(rows[2], rows[1]);
    tmp1 = shuffle<0, 0, 0, 2>(tmpa, tmpa);
    tmp2 = shuffle<0, 0, 0, 2>(tmpb, tmpb);
    tmp3 = shuffle<3, 3, 3, 3>(rows[2], rows[1]);
    cofactors[1] = tmp0 * tmp1 - tmp2 * tmp3;

    tmpa = shuffle<2, 2, 2, 2>(rows[3], rows[2]);
    tmpb = shuffle<1, 1, 1, 1>(rows[3], rows[2]);
    tmp0 = shuffle<1, 1, 1, 1>(rows[2], rows[1]);
    tmp1 = shuffle<0, 0, 0, 2>(tmpa, tmpa);
    tmp2 = shuffle<0, 0, 0, 2>(tmpb, tmpb);
    tmp3 = shuffle<2, 2, 2, 2>(rows[2], rows[1]);
    cofactors[2] = tmp0 * tmp1 - tmp2 * tmp3;

    tmpa = shuffle<3, 3, 3, 3>(rows[3], rows[2]);
    tmpb = shuffle<0, 0, 0, 0>(rows[3], rows[2]);
    tmp0 = shuffle<0, 0, 0, 0>(rows[2], rows[1]);
    tmp1 = shuffle<0, 0, 0, 2>(tmpa, tmpa);
    tmp2 = shuffle<0, 0, 0, 2>(tmpb, tmpb);
    tmp3 = shuffle<3, 3, 3, 3>(rows[2], rows[1]);
    cofactors[3] = tmp0 * tmp1 - tmp2 * tmp3;

    tmpa = shuffle<2, 2, 2, 2>(rows[3], rows[2]);
    tmpb = shuffle<0, 0, 0, 0>(rows[3], rows[2]);
    tmp0 = shuffle<0, 0, 0, 0>(rows[2], rows[1]);
    tmp1 = shuffle<0, 0, 0, 2>(tmpa, tmpa);
    tmp2 = shuffle<0, 0, 0, 2>(tmpb, tmpb);
    tmp3 = shuffle<2, 2, 2, 2>(rows[2], rows[1]);
    cofactors[4] = tmp0 * tmp1 - tmp2 * tmp3;

    tmpa = shuffle<1, 1, 1, 1>(rows[3], rows[2]);
    tmpb = shuffle<0, 0, 0, 0>(rows[3], rows[2]);
    tmp0 = shuffle<0, 0, 0, 0>(rows[2], rows[1]);
    tmp1 = shuffle<0, 0, 0, 2>(tmpa, tmpa);
    tmp2 = shuffle<0, 0, 0, 2>(tmpb, tmpb);
    tmp3 = shuffle<1, 1, 1, 1>(rows[2], rows[1]);
    cofactors[5] = tmp0 * tmp1 - tmp2 * tmp3;

    static float4 const& pmpm = float4(1.0f, -1.0f, 1.0f, -1.0f);
    static float4 const& mpmp = float4(-1.0f, 1.0f, -1.0f, 1.0f);

    float4 r01 = shuffle<0, 0, 0, 0>(rows[1], rows[0]);
    float4 v0 = shuffle<0, 2, 2, 2>(r01, r01);
    float4 r10 = shuffle<1, 1, 1, 1>(rows[1], rows[0]);
    float4 v1 = shuffle<0, 2, 2, 2>(r10, r10);
    r01 = shuffle<2, 2, 2, 2>(rows[1], rows[0]);
    float4 v2 = shuffle<0, 2, 2, 2>(r01, r01);
    r10 = shuffle<3, 3, 3, 3>(rows[1], rows[0]);
    float4 v3 = shuffle<0, 2, 2, 2>(r10, r10);

    float4 inv0 = mpmp * ((v1 * cofactors[0] - v2 * cofactors[1]) + v3 * cofactors[3]);
    float4 inv1 = pmpm * ((v0 * cofactors[0] - v2 * cofactors[3]) + v3 * cofactors[4]);
    float4 inv2 = mpmp * ((v0 * cofactors[1] - v1 * cofactors[3]) + v3 * cofactors[5]);
    float4 inv3 = pmpm * ((v0 * cofactors[2] - v1 * cofactors[4]) + v2 * cofactors[5]);
    float4 r = shuffle<0, 2, 0, 2>(shuffle<0, 0, 0, 0>(inv0, inv1), shuffle<0, 0, 0, 0>(inv2, inv3));

    float4 det = dot(rows[0], r);
    float4 recipr = simd::rcp<1>(det);

    rows[0] = inv0 * recipr;
    rows[1] = inv1 * recipr;
    rows[2] = inv2 * recipr;
    rows[3] = inv3 * recipr;
  }
private:
  simd::float4 rows[4];
};

inline Matrix< simd::float4 > operator*(Matrix< simd::float4 > const& m, Matrix< simd::float4 > const& n)
{
  using simd::float4;
  using simd::shuffle;

  Matrix< simd::float4 > result;
  for (size_t i = 0; i < 4; ++i)
  {
    float4 row = shuffle<0, 0, 0, 0>(m.row(i)) * n.row(0);
    row += shuffle<1, 1, 1, 1>(m.row(i)) * n.row(1);
    row += shuffle<2, 2, 2, 2>(m.row(i)) * n.row(2);
    row += shuffle<3, 3, 3, 3>(m.row(i)) * n.row(3);
    result.setRow(i, row);
  }
  return result;
}

inline vector< 4, simd::float4 > operator*(Matrix< simd::float4 > const& m, vector< 4, simd::float4 > const& v)
{
  Matrix< simd::float4 > tmp;
  tmp.setRow(0, v.x);
  tmp.setRow(1, v.y);
  tmp.setRow(2, v.z);
  tmp.setRow(3, v.w);
  Matrix< simd::float4 > res = m * tmp;
  return vector< 4, simd::float4 >(res.row(0), res.row(1), res.row(2), res.row(3));
}


inline vector< 3, simd::float4 > operator*(Matrix< simd::float4 > const& m, vector< 3, simd::float4 > const& v)
{
  vector< 4, simd::float4 > tmp(v[0], v[1], v[2], 1);
  vector< 4, simd::float4 > res = m * tmp;
  return vector< 3, simd::float4 >(res[0] / res[3], res[1] / res[3], res[2] / res[3]);
}


template < typename T >
inline std::ostream& operator<<(std::ostream& out, Matrix< T > const& m)
{
  out << m.row(0) << "\n";
  out << m.row(1) << "\n";
  out << m.row(2) << "\n";
  out << m.row(3) << "\n";
  return out;
}

} // math
} // virvo

