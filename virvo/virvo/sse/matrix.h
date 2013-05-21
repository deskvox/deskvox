#pragma once

#include "vec.h"
#include "vec3.h"
#include "vec4.h"

#include "../vvvecmath.h"

#include <ostream>

namespace virvo
{
namespace sse
{
class CACHE_ALIGN Matrix
{
public:
  inline Matrix()
  {
    identity();
  }

  inline Matrix(virvo::Matrix const& m)
  {
    CACHE_ALIGN float row1[4];
    CACHE_ALIGN float row2[4];
    CACHE_ALIGN float row3[4];
    CACHE_ALIGN float row4[4];

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
    CACHE_ALIGN float row1[4];
    CACHE_ALIGN float row2[4];
    CACHE_ALIGN float row3[4];
    CACHE_ALIGN float row4[4];

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
  }

  inline Vec row(size_t i) const
  {
    assert(i < 4);
    return rows[i];
  }

  inline void setRow(size_t i, Vec const& v)
  {
    assert(i < 4);
    rows[i] = v;
  }

  void identity()
  {
    rows[0] = Vec(1.0f, 0.0f, 0.0f, 0.0f);
    rows[1] = Vec(0.0f, 1.0f, 0.0f, 0.0f);
    rows[2] = Vec(0.0f, 0.0f, 1.0f, 0.0f);
    rows[3] = Vec(0.0f, 0.0f, 0.0f, 1.0f);
  }

  void transpose()
  {
    Vec tmp1 = _mm_unpacklo_ps(rows[0], rows[1]);
    Vec tmp2 = _mm_unpacklo_ps(rows[2], rows[3]);
    Vec tmp3 = _mm_unpackhi_ps(rows[0], rows[1]);
    Vec tmp4 = _mm_unpackhi_ps(rows[2], rows[3]);

    rows[0] = _mm_movelh_ps(tmp1, tmp2);
    rows[1] = _mm_movehl_ps(tmp2, tmp1);
    rows[2] = _mm_movelh_ps(tmp3, tmp4);
    rows[3] = _mm_movehl_ps(tmp4, tmp3);
  }

  void invert()
  {
    Vec cofactors[6];

    Vec tmpa = shuffle<3, 3, 3, 3>(rows[3], rows[2]);
    Vec tmpb = shuffle<2, 2, 2, 2>(rows[3], rows[2]);
    Vec tmp0 = shuffle<2, 2, 2, 2>(rows[2], rows[1]);
    Vec tmp1 = shuffle<0, 0, 0, 2>(tmpa, tmpa);
    Vec tmp2 = shuffle<0, 0, 0, 2>(tmpb, tmpb);
    Vec tmp3 = shuffle<3, 3, 3, 3>(rows[2], rows[1]);
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

    static Vec const& pmpm = Vec(1.0f, -1.0f, 1.0f, -1.0f);
    static Vec const& mpmp = Vec(-1.0f, 1.0f, -1.0f, 1.0f);

    Vec r01 = shuffle<0, 0, 0, 0>(rows[1], rows[0]);
    Vec v0 = shuffle<0, 2, 2, 2>(r01, r01);
    Vec r10 = shuffle<1, 1, 1, 1>(rows[1], rows[0]);
    Vec v1 = shuffle<0, 2, 2, 2>(r10, r10);
    r01 = shuffle<2, 2, 2, 2>(rows[1], rows[0]);
    Vec v2 = shuffle<0, 2, 2, 2>(r01, r01);
    r10 = shuffle<3, 3, 3, 3>(rows[1], rows[0]);
    Vec v3 = shuffle<0, 2, 2, 2>(r10, r10);

    Vec inv0 = mpmp * ((v1 * cofactors[0] - v2 * cofactors[1]) + v3 * cofactors[3]);
    Vec inv1 = pmpm * ((v0 * cofactors[0] - v2 * cofactors[3]) + v3 * cofactors[4]);
    Vec inv2 = mpmp * ((v0 * cofactors[1] - v1 * cofactors[3]) + v3 * cofactors[5]);
    Vec inv3 = pmpm * ((v0 * cofactors[2] - v1 * cofactors[4]) + v2 * cofactors[5]);
    Vec r = shuffle<0, 2, 0, 2>(shuffle<0, 0, 0, 0>(inv0, inv1), shuffle<0, 0, 0, 0>(inv2, inv3));

    Vec det = dot(rows[0], r);
    Vec rcp = fast::rcp<1>(det);

    rows[0] = inv0 * rcp;
    rows[1] = inv1 * rcp;
    rows[2] = inv2 * rcp;
    rows[3] = inv3 * rcp;
  }
private:
  Vec rows[4];
};

inline Matrix operator*(Matrix const& m, Matrix const& n)
{
  Matrix result;
  for (size_t i = 0; i < 4; ++i)
  {
    Vec row = shuffle<0, 0, 0, 0>(m.row(i)) * n.row(0);
    row += shuffle<1, 1, 1, 1>(m.row(i)) * n.row(1);
    row += shuffle<2, 2, 2, 2>(m.row(i)) * n.row(2);
    row += shuffle<3, 3, 3, 3>(m.row(i)) * n.row(3);
    result.setRow(i, row);
  }
  return result;
}

inline Vec3 operator*(Matrix const& m, Vec3 const& v)
{
  Vec3 result = v;
  return result;
}

inline Vec4 operator*(Matrix const& m, Vec4 const& v)
{
  Matrix tmp;
  tmp.setRow(0, v.x);
  tmp.setRow(1, v.y);
  tmp.setRow(2, v.z);
  tmp.setRow(3, v.w);
  Matrix res = m * tmp;
  return Vec4(res.row(0), res.row(1), res.row(2), res.row(3));
}

inline std::ostream& operator<<(std::ostream& out, Matrix const& m)
{
  out << m.row(0) << "\n";
  out << m.row(1) << "\n";
  out << m.row(2) << "\n";
  out << m.row(3) << "\n";
  return out;
}

} // sse
} // virvo

