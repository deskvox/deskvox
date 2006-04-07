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

#ifndef _VVECMATH_H_
#define _VVECMATH_H_

#include <float.h>

#include "vvexport.h"

//============================================================================
// Constant Definitions
//============================================================================

                                                  ///< compiler independent definition for pi
const float VV_PI = 3.1415926535897932384626433832795028841971693993751058f;
const float VV_FLT_MAX = FLT_MAX;                 ///< maximum float value

//============================================================================
// Forward Declarations
//============================================================================

class vvVector3;
class vvVector4;
class vvMatrix;

//============================================================================
// Class Definitions
//============================================================================

class VIRVOEXPORT vvVecmath
{
  public:
    static float sgn(float);
};

/** 4x4 matrix type.
 Matrix elements are: e[row][column]
 @author Jurgen P. Schulze (jschulze@ucsd.edu)
*/
class VIRVOEXPORT vvMatrix
{
  private:
    void LUDecomposition(int index[4], float &d);
    void LUBackSubstitution(int index[4], float b[4]);

  public:
    float e[4][4];                                ///< matrix elements: [row][column]

    vvMatrix();
    vvMatrix(const vvMatrix*);
    vvMatrix operator*(const vvMatrix) const;
    void print(const char*);
    void identity();
    void zero();
    void translate(float, float, float);
    void translate(const vvVector3*);
    void scale(float, float, float);
    void scale(float);
    vvMatrix rotate(float, float, float, float);
    vvMatrix rotate(float, const vvVector3*);
    void multiplyPre(const vvMatrix*);
    void multiplyPost(const vvMatrix*);
    void transpose();
    float diagonal();
    void invertOrtho();
    void invert2D();
    void copy(const vvMatrix*);
    void copyTrans(const vvMatrix*);
    void copyRot(const vvMatrix*);
    void transOnly();
    void rotOnly();
    void killTrans();
    void killRot();
    bool equal(const vvMatrix*);
    void makeGL(float*);
    void getGL(float*);
    void getGL(double*);
    void get(float*);
    void set(float*);
    void get(double*);
    void set(double*);
    void setRow(int, float, float, float, float);
    void setRow(int, vvVector3*);
    void setColumn(int, float, float, float, float);
    void setColumn(int, vvVector3*);
    void setColumn(int, vvVector3&);
    void getRow(int, float*, float*, float*, float*);
    void getRow(int, vvVector3*);
    void getColumn(int, float*, float*, float*, float*);
    void getColumn(int, vvVector3*);
    void random(int, int);
    void random(float, float);
    void invert();
    void swapRows(int, int);
    void swapColumns(int, int);
    void setProjOrtho(float, float, float, float, float, float);
    void getProjOrtho(float*, float*, float*, float*, float*, float*);
    void setProjPersp(float, float, float, float, float, float);
    void getProjPersp(float*, float*, float*, float*, float*, float*);
    bool isProjOrtho();
    void makeLookAt(float, float, float, float, float, float, float, float, float);
    float getNearPlaneZ();
    vvMatrix trackballRotation(int, int, int, int, int, int);
    void computeEulerAngles(float*, float*, float*);
};

/** 3D vector primitive.
 @author Juergen Schulze-Doebold (schulze@hlrs.de)
*/
class VIRVOEXPORT vvVector4
{
  public:
    float e[4];                                   ///< vector elements (x|y|z|w)

    vvVector4();
    vvVector4(const vvVector4*);
    float &operator[](int);
    float operator[](int) const;
    void set(float, float, float, float);
    void multiply(const vvMatrix*);
    void copy(const vvVector4*);
    void print(const char*);
};

/** 3D vector primitive, also used for points
 @author Jurgen P. Schulze (jschulze@ucsd.edu)
*/
class VIRVOEXPORT vvVector3
{
  public:
    float e[3];                                   ///< vector elements (x|y|z)

    vvVector3();
    vvVector3(float, float, float);
    vvVector3(const vvVector3*);
    vvVector3 operator^(const vvVector3) const;
    float &operator[](int);
    float operator[](int) const;
    void  set(float, float, float);
    void  get(float*, float*, float*);
    void  copy(const vvVector3*);
    void  copy(const vvVector3&);
    void  copy(const vvVector4*);
    void  add(const vvVector3*);
    void  add(float);
    void  add(float, float, float);
    void  sub(const vvVector3*);
    void  sub(float);
    void  scale(float);
    void  scale(const vvVector3*);
    void  scale(float, float, float);
    float dot(const vvVector3*) const;
    float angle(vvVector3*);
    void  cross(const vvVector3*);
    void  multiply(const vvMatrix*);
    float distance(const vvVector3*);
    float length() const;
    void  planeNormalPPV(const vvVector3*, const vvVector3*, const vvVector3*);
    float distPointPlane(const vvVector3*, const vvVector3*) const;
    void  normalize();
    void  negate();
    bool  equal(const vvVector3*);
    void  random(int, int);
    void  random(float, float);
    void  random(double, double);
    void  print(const char*);
    void  getRow(const vvMatrix*, int);
    void  getColumn(const vvMatrix*, int);
    void  swap(vvVector3*);
    bool  isectPlaneLine(const vvVector3*, const vvVector3*, const vvVector3*, const vvVector3*);
    bool  isectPlaneRay(const vvVector3*, const vvVector3*, const vvVector3*, const vvVector3*);
    int   isectPlaneCuboid(const vvVector3*, const vvVector3*, const vvVector3*, const vvVector3*);
    int   isectRayCylinder(const vvVector3*, const vvVector3*, float, const vvVector3*, const vvVector3*);
    bool  isectRayTriangle(const vvVector3*, const vvVector3*, const vvVector3*, const vvVector3*, const vvVector3*);
    float isectLineLine(const vvVector3&, const vvVector3&, const vvVector3&, const vvVector3&);
    bool  isSameSideLine2D(const vvVector3*, const vvVector3*, const vvVector3*, const vvVector3*);
    bool  isInTriangle(const vvVector3*, const vvVector3*, const vvVector3*);
    void  cyclicSort(int, const vvVector3*);
    void  zero();
    bool  isZero();
    void  getSpherical(float*, float*, float*);
    void  directionCosines(const vvVector3*);
    static float signum(float);
    vvVector3 operator + (const vvVector3 &other);
    vvVector3 operator - (const vvVector3 &other);
    vvVector3 operator * (const vvVector3 &other);

    vvVector3 operator * (const float scalar);
    friend vvVector3 operator * (const float scalar, const vvVector3 &other);

    vvVector3& operator = (const vvVector3 &other);
    vvVector3& operator += (const vvVector3 &other);
    vvVector3& operator -= (const vvVector3 &other);

    vvVector3 operator + (void) const;
    vvVector3 operator - (void) const;
};

/** 3D plane primitive.
 @author Jurgen Schulze (jschulze@ucsd.edu)
*/
class vvPlane
{
  public:
    vvVector3 _point;
    vvVector3 _normal;

    vvPlane(const vvVector3&, const vvVector3&);
    vvPlane(const vvVector3&, const vvVector3&, const vvVector3&);
    bool isSameSide(const vvVector3&, const vvVector3&) const;
    float dist(const vvVector3&) const;
};
#endif

// EOF
