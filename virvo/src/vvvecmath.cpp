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

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "vvvirvo.h"
#include "vvvecmath.h"

#ifdef __sun
#define sinf sin
#define cosf cos
#define acosf acos
#define sqrtf sqrt
#define expf exp
#define atanf atan
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

using namespace std;

//============================================================================
// vvVecmath Class Methods
//============================================================================

//----------------------------------------------------------------------------
/** Signum function.
  @param a number to test for sign
  @return 1 if a is positive or zero, -1 if a is negative
*/
float vvVecmath::sgn(float a)
{
  if (a>=0) return 1.0f;
  else return -1.0f;
}

//============================================================================
// vvMatrix Class Methods
//============================================================================

//----------------------------------------------------------------------------
/// Constructor for empty matrix (all values 0.0)
vvMatrix::vvMatrix()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = 0.0;
}

//----------------------------------------------------------------------------
/// Copy constructor.
vvMatrix::vvMatrix(const vvMatrix* m)
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = m->e[row][col];
}

//----------------------------------------------------------------------------
/** Multiplication. Operands will be multiplied from left to right.
 */
vvMatrix vvMatrix::operator*(const vvMatrix operand) const
{
  vvMatrix tmp;
  tmp = this;
  tmp.multiplyPost(&operand);
  return tmp;
}

//----------------------------------------------------------------------------
/// Print the matrix to stdout for debugging
void vvMatrix::print(const char* title)
{
  int row, col;

  cerr << title << endl;

  cerr.setf(ios::fixed, ios::floatfield);
  cerr.precision(3);

  for (row=0; row<4; ++row)
  {
    for (col=0; col<4; ++col)
      cerr << setw(10) << e[row][col];
    cerr << endl;
  }
}

//----------------------------------------------------------------------------
/// Set identity matrix (diagonal=1, rest=0)
void vvMatrix::identity()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      if (row==col) e[row][col] = 1.0;
  else e[row][col] = 0.0;
}

//----------------------------------------------------------------------------
/// Set all matrix elements to zero
void vvMatrix::zero()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = 0.0;
}

//----------------------------------------------------------------------------
/// Apply a translation.
void vvMatrix::translate(float x, float y, float z)
{
  e[0][3] += x;
  e[1][3] += y;
  e[2][3] += z;
}

//----------------------------------------------------------------------------
/// Apply a translation.
void vvMatrix::translate(const vvVector3* v)
{
  e[0][3] += v->e[0];
  e[1][3] += v->e[1];
  e[2][3] += v->e[2];
}

//----------------------------------------------------------------------------
/// Apply a non-uniform scale.
void vvMatrix::scale(float x, float y, float z)
{
  vvMatrix s;                                     // scaling matrix

  s.e[0][0] = x;
  s.e[1][1] = y;
  s.e[2][2] = z;
  s.e[3][3] = 1.0f;
  this->multiplyPre(&s);
}

//----------------------------------------------------------------------------
/** Apply a uniform scale.
  @param a scale factor
*/
void vvMatrix::scale(float a)
{
  vvMatrix s;                                     // scaling matrix

  s.e[0][0] = a;
  s.e[1][1] = a;
  s.e[2][2] = a;
  s.e[3][3] = 1.0f;
  this->multiplyPre(&s);
}

//----------------------------------------------------------------------------
/** Rotation about vector x/y/z by angle a (radian).
 (Source: Foley et.al. page 227)
 @return rotation matrix
*/
vvMatrix vvMatrix::rotate(float a, float x, float y, float z) 
{
  vvMatrix rot;                                   // rotation matrix
  float cosfa, sinfa;                             // shortcuts
  float d;                                        // divisor

  // normalize vector:
  d = (float)(sqrt(x * x + y * y + z * z));
  if (d == 0.0) 
  {
    cerr << "vvMatrix::rotate: invalid rotation vector" << endl;
    return rot;                       // division by zero error
  }
  x /= d;
  y /= d;
  z /= d;

  // Precompute recurring values:
  cosfa = cosf(a);
  sinfa = sinf(a);

  // Compute rotation matrix:
  rot.e[0][0] = x * x + cosfa * (1 - x * x);
  rot.e[0][1] = x * y * (1 - cosfa) - z * sinfa;
  rot.e[0][2] = x * z * (1 - cosfa) + y * sinfa;
  rot.e[0][3] = 0.0;

  rot.e[1][0] = x * y * (1 - cosfa) + z * sinfa;
  rot.e[1][1] = y * y + cosfa * (1 - y * y);
  rot.e[1][2] = y * z * (1 - cosfa) - x * sinfa;
  rot.e[1][3] = 0.0;

  rot.e[2][0] = x * z * (1 - cosfa) - y * sinfa;
  rot.e[2][1] = y * z * (1 - cosfa) + x * sinfa;
  rot.e[2][2] = z * z + cosfa * (1 - z * z);
  rot.e[2][3] = 0.0;

  rot.e[3][0] = 0.0;
  rot.e[3][1] = 0.0;
  rot.e[3][2] = 0.0;
  rot.e[3][3] = 1.0;

  // Perform rotation:
  multiplyPre(&rot);
  return rot;
}

//----------------------------------------------------------------------------
/** Rotation about vector v by angle a (radian).
  @return rotation matrix
 */
vvMatrix vvMatrix::rotate(float a, const vvVector3* v)
{
  return vvMatrix::rotate(a, v->e[0], v->e[1], v->e[2]);
}

//----------------------------------------------------------------------------
/** Multiplies two matrices. If matrices are resolved from left to right,
  this operation would be: this = m * this
*/
void vvMatrix::multiplyPre(const vvMatrix* m)
{
  int row, col;
  vvMatrix bak(this);                             // backup of current matrix

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = bak.e[row][0] * m->e[0][col] + bak.e[row][1] * m->e[1][col] +
        bak.e[row][2] * m->e[2][col] + bak.e[row][3] * m->e[3][col];
}

//----------------------------------------------------------------------------
/** Multiplies two matrices. If matrices are resolved from left to right,
  this operation would be: this = this * m
*/
void vvMatrix::multiplyPost(const vvMatrix* m)
{
  int row, col;
  vvMatrix bak(this);                             // backup of current matrix

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = bak.e[0][col] * m->e[row][0] + bak.e[1][col] * m->e[row][1] +
        bak.e[2][col] * m->e[row][2] + bak.e[3][col] * m->e[row][3];
}

//----------------------------------------------------------------------------
/** Inverts an _orthogonal_ matrix
 Orthogonal means: columns and rows are perpenducular to each other and are unit vectors
*/
void vvMatrix::invertOrtho()
{
  int row, col;
  vvMatrix bak(this);                             // backup of current matrix

  for (row=0; row<3; ++row)
    for (col=0; col<3; ++col)
      e[row][col] = bak.e[col][row];
}

//----------------------------------------------------------------------------
/// Inverts _only_ the 2D part of a matrix
void vvMatrix::invert2D()
{
  vvMatrix bak(this);                             // backup of current matrix
  float factor;                                   // constant factor
  float det;                                      // 2D part determinant

  det = e[0][0] * e[1][1] - e[0][1] * e[1][0];
  if (det == 0.0) return;                         // determinant zero error
  factor = 1.0f / det;

  e[0][0] =  factor * bak.e[1][1];
  e[0][1] = -factor * bak.e[0][1];
  e[1][0] = -factor * bak.e[1][0];
  e[1][1] =  factor * bak.e[0][0];
}

//----------------------------------------------------------------------------
/// Transposes a matrix (=mirror at diagonal)
void vvMatrix::transpose()
{
  int row, col;
  vvMatrix bak(this);                             // backup of current matrix

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = bak.e[col][row];
}

//----------------------------------------------------------------------------
/// Returns the multiplied diagonal values
float vvMatrix::diagonal()
{
  float mult = 1.0f;
  int i;

  for (i=0; i<4; ++i)
    mult *= e[i][i];
  return mult;
}

//----------------------------------------------------------------------------
/// Creates an identical copy of a matrix
void vvMatrix::copy(const vvMatrix* m)
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = m->e[row][col];
}

//----------------------------------------------------------------------------
/// Copies only the translational part of a matrix and keeps the rest untouched
void vvMatrix::copyTrans(const vvMatrix* m)
{
  e[0][3] = m->e[0][3];
  e[1][3] = m->e[1][3];
  e[2][3] = m->e[2][3];
}

//----------------------------------------------------------------------------
/// Copies only the rotational part of a matrix and keeps the rest untouched
void vvMatrix::copyRot(const vvMatrix* m)
{
  int row, col;

  for (row=0; row<3; ++row)
    for (col=0; col<3; ++col)
      e[row][col] = m->e[row][col];
}

//----------------------------------------------------------------------------
/// Make pure translational matrix.
void vvMatrix::transOnly()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      if (row==col) e[row][col] = 1.0f;
  else if (col!=3) e[row][col] = 0.0f;
}

//----------------------------------------------------------------------------
/// Make pure rotational matrix.
void vvMatrix::rotOnly()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      if (row==3 || col==3)
      {
        if (row==3 && col==3) e[row][col] = 1.0f;
        else e[row][col] = 0.0f;
      }
}

//----------------------------------------------------------------------------
/// Overwrites the translational part of a matrix with 0's
void vvMatrix::killTrans()
{
  e[0][3] = e[1][3] = e[2][3] = 0.0;
}

//----------------------------------------------------------------------------
/// Overwrites the rotational part of a matrix with 0's
void vvMatrix::killRot()
{
  int row, col;

  for (row=0; row<3; ++row)
    for (col=0; col<3; ++col)
      e[row][col] = 0.0;
}

//----------------------------------------------------------------------------
/// Compares two matrices. Returns true if equal, otherwise false
bool vvMatrix::equal(const vvMatrix* m)
{
  bool result = true;
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      if (e[row][col] != m->e[row][col]) result = false;
  return result;
}

//----------------------------------------------------------------------------
/** Stores the matrix elements in a linear array in column-major order.
  Column-Major order means (if matrix is written with translational
  part on the right side):
  <PRE>
  e0  e4  e8  e12
  e1  e5  e9  e13
  e2  e6  e10 e14
  e3  e7  e11 e15
  </PRE>
 Important: the array pointer must provide space for 16 float values!<BR>
 This function is needed to convert the matrix values to the OpenGL
format used e.g. in the glLoadMatrixf() command.
@see getGL
*/
void vvMatrix::makeGL(float* array)
{
  int row, col, i=0;

  for (col=0; col<4; ++col)
    for (row=0; row<4; ++row)
      array[i++] = e[row][col];
}

//----------------------------------------------------------------------------
/** Converts an OpenGL matrix to the vecmath matrix format
  @see makeGL
*/
void vvMatrix::getGL(float* glmatrix)
{
  int row, col, i=0;

  for (col=0; col<4; ++col)
    for (row=0; row<4; ++row)
      e[row][col] = glmatrix[i++];
}

//----------------------------------------------------------------------------
/** @see getGL
 */
void vvMatrix::getGL(double* glmatrix)
{
  int i;
  float mat[16];

  for (i=0; i<16; ++i)
  {
    mat[i] = float(glmatrix[i]);
  }
  getGL(mat);
}

//----------------------------------------------------------------------------
/** Returns the matrix in float format.
 A float array with space for 16 float elements must be given!
*/
void vvMatrix::get(float* elements)
{
  int col, row, i=0;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
  {
    elements[i] = e[row][col];
    ++i;
  }
}

//----------------------------------------------------------------------------
/** Copies the matrix from a source which is in float format.
 A float array with 16 float elements must be given!
*/
void vvMatrix::set(float* elements)
{
  int col, row, i=0;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
  {
    e[row][col] = elements[i];
    ++i;
  }
}

//----------------------------------------------------------------------------
/** Returns the matrix in double format.
 A double array with space for 16 double elements must be given!
*/
void vvMatrix::get(double* elements)
{
  int col, row, i=0;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
  {
    elements[i] = (double)e[row][col];
    ++i;
  }
}

//----------------------------------------------------------------------------
/** Copies the matrix from a source which is in double format.
 A double array with 16 double elements must be given!
*/
void vvMatrix::set(double* elements)
{
  int col, row, i=0;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
  {
    e[row][col] = (float)elements[i];
    ++i;
  }
}

//----------------------------------------------------------------------------
/** Set matrix values for a specific row.
  @param row      row index
  @param a,b,c,d  new values (left to right)
*/
void vvMatrix::setRow(int row, float a, float b, float c, float d)
{
  e[row][0] = a;
  e[row][1] = b;
  e[row][2] = c;
  e[row][3] = d;
}

//----------------------------------------------------------------------------
/** Set matrix values of a specific row from a vector.
  The rightmost matrix element of the row is not changed.
  @param row  row index
  @param vec  vector with new elements
*/
void vvMatrix::setRow(int row, vvVector3* vec)
{
  e[row][0] = vec->e[0];
  e[row][1] = vec->e[1];
  e[row][2] = vec->e[2];
}

//----------------------------------------------------------------------------
/** Set matrix values for a specific column.
  @param col      column index
  @param a,b,c,d  new values (top to bottom)
*/
void vvMatrix::setColumn(int col, float a, float b, float c, float d)
{
  e[0][col] = a;
  e[1][col] = b;
  e[2][col] = c;
  e[3][col] = d;
}

//----------------------------------------------------------------------------
/** Set matrix values of a specific column from a vector.
  The bottom matrix element of the column is not changed.
  @param col  column index
  @param vec  vector with new elements
*/
void vvMatrix::setColumn(int col, vvVector3* vec)
{
  e[0][col] = vec->e[0];
  e[1][col] = vec->e[1];
  e[2][col] = vec->e[2];
}

//----------------------------------------------------------------------------
/** @see setColumn
 */
void vvMatrix::setColumn(int col, vvVector3& vec)
{
  e[0][col] = vec[0];
  e[1][col] = vec[1];
  e[2][col] = vec[2];
}

//----------------------------------------------------------------------------
/** Get matrix values of a specific row.
  @param row row index
  @return Row values are found in a,b,c,d (left to right)
*/
void vvMatrix::getRow(int row, float* a, float* b, float* c, float* d)
{
  *a = e[row][0];
  *b = e[row][1];
  *c = e[row][2];
  *d = e[row][3];
}

//----------------------------------------------------------------------------
/** Get matrix values of a specific row and store them in a vector.
  The rightmost matrix element of the row is ignored.
  @param row  row index
  @param vec  vector to obtain matrix elements
*/
void vvMatrix::getRow(int row, vvVector3* vec)
{
  vec->e[0] = e[row][0];
  vec->e[1] = e[row][1];
  vec->e[2] = e[row][2];
}

//----------------------------------------------------------------------------
/** Get matrix values of a specific column.
  @param col column index
  @return Column values are found in a,b,c,d (top to bottom)
*/
void vvMatrix::getColumn(int col, float* a, float* b, float* c, float* d)
{
  *a = e[0][col];
  *b = e[1][col];
  *c = e[2][col];
  *d = e[3][col];
}

//----------------------------------------------------------------------------
/** Get matrix values of a specific column and store them in a vector.
  The bottom matrix element of the column is ignored.
  @param col  column index
  @param vec  vector to obtain matrix elements
*/
void vvMatrix::getColumn(int col, vvVector3* vec)
{
  vec->e[0] = e[0][col];
  vec->e[1] = e[1][col];
  vec->e[2] = e[2][col];
}

//----------------------------------------------------------------------------
/// Creates matrix with random integer numbers in range from..to
void vvMatrix::random(int from, int to)
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = (float)(from + (rand() % (to - from + 1)));
}

//----------------------------------------------------------------------------
/// Creates matrix with random float numbers in range from..to
void vvMatrix::random(float from, float to)
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = (float)from + ((float)rand()/(float)RAND_MAX * (float)(to - from));
}

//-----------------------------------------------------------------------
/** LU decomposition.
  From  Numerical Recipes, p. 46.<BR>
  Given a matrix a[n][n], this routine replaces the matrix by the
  LU decomposition of a rowwise permutation of itself. n and a are
  input. a is output. indx[n] is an output vector that records
  the row permutation effected by the partial pivoting; d is output
  +-1 depending on wether the number of eow interchanges was even
  or odd, respectively.
  @author Daniel Weiskopf
  @see LUBackSubstitution
*/
void vvMatrix::LUDecomposition(int index[4], float &d)
{
  const float TINY = 1.0e-20f;
  const int n = 4;                                // Special case for n=4, could be any positive number
  int i, imax, j, k;
  float big, dum, sum, temp;
  float* vv = new float[n];                       // Stores the implicit scaling of each row

  d = 1.0f;

  // Loop over rows to get the implicit scaling information
  for(i = 0; i < n; i++)
  {
    big = 0.0f;
    for (j = 0; j < n; j++)
    {
      if ((temp = (float) fabs(e[i][j])) > big)
        big = temp;
    }
    if (big == 0.0)
      cerr << "Singular matrix in routine LUdcmp " << e[i][0] << " "
        << e[i][1] << " " << e[i][2] << " " << e[i][3];
    vv[i] = 1.0f / big;                           // Save the scaling
  }

  // Loop over columns for Crout's method
  for (j = 0; j < n; j++)
  {
    for (i = 0; i < j; i++)
    {
      sum = e[i][j];
      for (k = 0; k < i; k++)
        sum -= e[i][k] * e[k][j];
      e[i][j] = sum;
    }

    // Finds the pivot point
    big = 0.0f;
    imax = 0;
    for (i = j; i < n; i++)
    {
      sum = e[i][j];
      for (k = 0; k < j; k++)
        sum -= e[i][k] * e[k][j];
      e[i][j] = sum;
      if ((dum = vv[i] * (float) fabs(sum)) >= big)
      {
        big = dum;
        imax = i;
      }
    }

    // Do we need to interchange rows?
    if (j != imax)
    {
      for (k = 0; k < n; k++)
      {
        dum = e[imax][k];
        e[imax][k] = e[j][k];
        e[j][k] = dum;
      }
      d = -d;
      vv[imax] = vv[j];                           // Also interchange the scale factor
    }
    index[j] = imax;
    if (e[j][j] == 0.0)
      e[j][j] = TINY;
    if (j != n)
    {
      dum = 1/(e[j][j]);
      for (i = j+1; i < n; i++)
        e[i][j] *= dum;
    }
  }

  delete[] vv;
}

//----------------------------------------------------------------------------
/** LU backsubstitution.
  @author Daniel Weiskopf
  @see LUDecomposition
*/
void vvMatrix::LUBackSubstitution(int index[4], float b[4])
{
  int   i, ip, j;
  int   ii = -1;
  float sum;

  // Special case for n=4, could be any positive number
  int n = 4;

  for (i = 0; i < n; i++)
  {
    ip    = index[i];
    sum   = b[ip];
    b[ip] = b[i];
    if (ii >= 0)
      for (j = ii; j <= i-1; j++)
        sum -= e[i][j] * b[j];
    else if (sum)
      ii = i;
    b[i] = sum;
  }

  for (i = n-1; i >= 0; i--)
  {
    sum = b[i];
    for (j = i+1; j < n; j++)
      sum -= e[i][j] * b[j];
    b[i] = sum/e[i][i];
  }
}

//----------------------------------------------------------------------------
/** Invert a matrix (according to Numerical Recipes, p. 48)
  @author Daniel Weiskopf
*/
void vvMatrix::invert(void)
{
  float    d;
  int      index[4];
  float    col[4];
  float    y[4][4];
  int      i;
  int      j;

  LUDecomposition(index, d);
  for ( j = 0; j < 4; j++)
  {
    for ( i = 0; i < 4; i++)
      col[i] = 0.0f;
    col[j] = 1.0f;
    LUBackSubstitution(index, col);
    for ( i = 0; i < 4; i++)
      y[i][j] = col[i];
  }

  for ( j = 0; j < 4; j++)
    for ( i = 0; i < 4; i++)
      e[i][j] = y[i][j];
}

//----------------------------------------------------------------------------
/** Swap two matrix rows
  @param row1, row2  rows to swap [0..3]
*/
void vvMatrix::swapRows(int row1, int row2)
{
  int col;
  float buffer;

  for (col=0; col<4; ++col)
  {
    buffer = e[row1][col];
    e[row1][col] = e[row2][col];
    e[row2][col] = buffer;
  }
}

//----------------------------------------------------------------------------
/** Swap two matrix columns
  @param col1, col2 columns to swap [0..3]
*/
void vvMatrix::swapColumns(int col1, int col2)
{
  int row;
  float buffer;

  for (row=0; row<4; ++row)
  {
    buffer = e[row][col1];
    e[row][col1] = e[row][col2];
    e[row][col2] = buffer;
  }
}

//----------------------------------------------------------------------------
/// Set OpenGL compatible projection matrix for orthogonal projection
void vvMatrix::setProjOrtho(float left, float right, float bottom, float top,
float nearPlane, float farPlane)
{
  e[0][0] = 2.0f / (right - left);
  e[0][1] = 0.0f;
  e[0][2] = 0.0f;
  e[0][3] = (left + right) / (left - right);
  e[1][0] = 0.0f;
  e[1][1] = 2.0f / (top - bottom);
  e[1][2] = 0.0f;
  e[1][3] = (bottom + top) / (bottom - top);
  e[2][0] = 0.0f;
  e[2][1] = 0.0f;
  e[2][2] = 2.0f / (nearPlane - farPlane);
  e[2][3] = (nearPlane + farPlane) / (nearPlane - farPlane);
  e[3][0] = 0.0f;
  e[3][1] = 0.0f;
  e[3][2] = 0.0f;
  e[3][3] = 1.0f;
}

//----------------------------------------------------------------------------
/// Get parameters of OpenGL parallel projection matrix
void vvMatrix::getProjOrtho(float* left, float* right, float* bottom, float* top,
float* nearPlane, float* farPlane)
{
  *left       = 2.0f / (e[0][0] * ( (e[0][3] - 1.0f) / (e[0][3] + 1.0f) - 1.0f ));
  *right      = (*left) + 2.0f / e[0][0];
  *bottom     = 2.0f / (e[1][1] * ( (e[1][3] - 1.0f) / (e[1][3] + 1.0f) - 1.0f ));
  *top        = (*bottom) + 2.0f / e[1][1];
  *nearPlane  = 2.0f / (e[2][2] * ( (1.0f - e[2][3]) / (1.0f + e[2][3]) + 1.0f ));
  *farPlane   = (*nearPlane) - 2.0f / e[2][2];
}

//----------------------------------------------------------------------------
/// Set OpenGL compatible projection matrix for perspective projection
void vvMatrix::setProjPersp(float left, float right, float bottom, float top,
float nearPlane, float farPlane)
{
  e[0][0] = (nearPlane + nearPlane)  / (right - left);
  e[0][1] = 0.0f;
  e[0][2] = (right + left) / (right - left);
  e[0][3] = 0.0f;
  e[1][0] = 0.0f;
  e[1][1] = (nearPlane + nearPlane)  / (top - bottom);
  e[1][2] = (top + bottom) / (top - bottom);
  e[1][3] = 0.0f;
  e[2][0] = 0.0f;
  e[2][1] = 0.0f;
  e[2][2] = (farPlane + nearPlane) / (nearPlane - farPlane);
  e[2][3] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
  e[3][0] = 0.0f;
  e[3][1] = 0.0f;
  e[3][2] = -1.0f;
  e[3][3] = 0.0f;
}

//----------------------------------------------------------------------------
/// Get parameters of OpenGL perspective projection matrix
void vvMatrix::getProjPersp(float* left, float* right, float* bottom, float* top,
float* nearPlane, float* farPlane)
{
  *nearPlane  = e[2][3] / 2.0f * ((e[2][2] + 1.0f) / (e[2][2] - 1.0f) - 1.0f);
  *farPlane   = e[2][3] * (*nearPlane) / (2.0f * (*nearPlane) + e[2][3]);
  *left       = 2.0f * (*nearPlane) / (e[0][0] * ((e[0][2] + 1.0f) / (e[0][2] - 1.0f) - 1.0f));
  *right      = (*left) * (e[0][2] + 1.0f) / (e[0][2] - 1.0f);
  *bottom     = 2.0f * (*nearPlane) / (e[1][1] * ((e[1][2] + 1.0f) / (e[1][2] - 1.0f) - 1.0f));
  *top        = (*bottom) * (e[1][2] + 1.0f) / (e[1][2] - 1.0f);
}

//----------------------------------------------------------------------------
/** Checks if the matrix could be used as an orthogonal projection matrix.
  Strategy: Given a matrix m:<PRE>
    A0 A1 A2 A3
    B0 B1 B2 B3
    C0 C1 C2 C3
    D0 D1 D2 D3
  </PRE>
  The distinguishing matrix elements are D0, D1, and D2. They are called
  perspective scale factors.<BR>
  D0 is the perspective scale factor for the X axis,<BR>
  D1 is the perspective scale factor for the Y axis,<BR>
D2 is the perspective scale factor for the Z axis.<BR>
If all scale factors are zero, an orthogonal projection is used.
@return true if matrix is an orthogonal projection matrix, otherwise false
*/
bool vvMatrix::isProjOrtho()
{
  if (e[3][0]==0.0f && e[3][1]==0.0f && e[3][2]==0.0f) return true;
  else return false;
}

//-----------------------------------------------------------------------------
/** This function works exactly the same as gluLookAt: it creates a viewing
  matrix derived from an eye point, a reference point indicating the center
  of the scene, and an UP vector.
*/
void vvMatrix::makeLookAt(float eyeX, float eyeY, float eyeZ,
float centerX, float centerY, float centerZ,
float upX, float upY, float upZ)
{
  vvVector3 f, up, s, u, center, eye;

  center[0] = centerX;
  center[1] = centerY;
  center[2] = centerZ;
  eye[0] = eyeX;
  eye[1] = eyeY;
  eye[2] = eyeZ;
  f = center - eye;
  f.normalize();
  up[0] = upX;
  up[1] = upY;
  up[2] = upZ;
  up.normalize();
  s = f ^ up;
  u = s ^ f;

  identity();
  e[0][0] =  s[0];
  e[0][1] =  u[0];
  e[0][2] = -f[0];
  e[1][0] =  s[1];
  e[1][1] =  u[1];
  e[1][2] = -f[1];
  e[2][0] =  s[2];
  e[2][1] =  u[2];
  e[2][2] = -f[2];

  vvMatrix trans, result;
  trans.identity();
  eye = -eye;
  trans.setColumn(3, eye);
  *this = (*this) * trans;
}

//-----------------------------------------------------------------------------
/** Determine Z coordinate of near plane from glFrustum generated projection matrices.
  @return z coordinate of near plane
*/
float vvMatrix::getNearPlaneZ()
{
  return e[2][3] / 2.0f * ((e[2][2] + 1.0f) / (e[2][2] - 1.0f) - 1.0f);
}

//-----------------------------------------------------------------------------
/** Rotates the matrix according to a fictitious trackball, placed in
    the middle of the given window.
    The trackball is approximated by a Gaussian curve.
    The trackball coordinate system is: x=right, y=up, z=to viewer<BR>
    The origin of the mouse coordinates zero (0,0) is considered to be top left.
  @param width, height  window size in pixels
  @param fromX, fromY   mouse starting position in pixels
  @param toX, toY       mouse end position in pixels
*/
vvMatrix vvMatrix::trackballRotation(int width, int height, int fromX, int fromY, int toX, int toY)
{
  const float TRACKBALL_SIZE = 1.3f;              // virtual trackball size (empirical value)
  vvMatrix mInv;                                  // inverse of ObjectView matrix
  vvVector3 v1, v2;                               // mouse drag positions in normalized 3D space
  float smallSize;                                // smaller window size between width and height
  float halfWidth, halfHeight;                    // half window sizes
  float angle;                                    // rotational angle
  float d;                                        // distance

  // Compute mouse coordinates in window and normalized to -1..1
  // ((0,0)=window center, (-1,-1) = bottom left, (1,1) = top right)
  halfWidth   = (float)width  / 2.0f;
  halfHeight  = (float)height / 2.0f;
  smallSize   = (halfWidth < halfHeight) ? halfWidth : halfHeight;
  v1.e[0]     = ((float)fromX - halfWidth)  / smallSize;
  v1.e[1]     = ((float)(height-fromY) - halfHeight) / smallSize;
  v2.e[0]     = ((float)toX   - halfWidth)  / smallSize;
  v2.e[1]     = ((float)(height-toY)   - halfHeight) / smallSize;

  // Compute z-coordinates on Gaussian trackball:
  d       = sqrtf(v1.e[0] * v1.e[0] + v1.e[1] * v1.e[1]);
  v1.e[2] = expf(-TRACKBALL_SIZE * d * d);
  d       = sqrtf(v2.e[0] * v2.e[0] + v2.e[1] * v2.e[1]);
  v2.e[2] = expf(-TRACKBALL_SIZE * d * d);

  // Compute rotational angle:
  angle = v1.angle(&v2);                          // angle = angle between v1 and v2

  // Compute rotational axis:
  v2.cross(&v1);                                  // v2 = v2 x v1 (cross product)

  // Convert axis coordinates (v2) from WCS to OCS:
  mInv.identity();
  mInv.copyRot(this);                             // copy rotational part of mv to mInv
  mInv.invertOrtho();                             // invert orthogonal matrix mInv
  v2.multiply(&mInv);                             // v2 = v2 x mInv (matrix multiplication)
  v2.normalize();                                 // normalize v2 before rotation

  // Perform acutal model view matrix modification:
  return rotate(-angle, v2.e[0], v2.e[1], v2.e[2]);      // rotate model view matrix
}

/** Compute Euler angles for a matrix. The angles are returned in Radians.
  Source: http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q37
*/
void vvMatrix::computeEulerAngles(float* angleX, float* angleY, float* angleZ)
{
  float tx, ty, c;

  *angleY = asinf(e[2][0]);                       // Calculate Y-axis angle
  c =  cosf(*angleY);
  if (fabs( c ) > 0.005f)                         // Gimball lock?
  {
    tx      =  e[2][2] / c;                       // No, so get X-axis angle
    ty      = -e[2][1] / c;
    *angleX = atan2f(ty, tx);
    tx      =  e[0][0] / c;                       // Get Z-axis angle
    ty      = -e[1][0] / c;
    *angleZ = atan2f( ty, tx );
  }
  else                                            // Gimball lock has occurred
  {
    *angleX = 0.0f;                               // Set X-axis angle to zero
    tx      = e[1][1];                            // And calculate Z-axis angle
    ty      = e[0][1];
    *angleZ = atan2f( ty, tx );
  }

  // Return only positive angles in [0, 2*VV_PI]:
  if (*angleX < 0.0f) *angleX += 2.0f * VV_PI;
  if (*angleY < 0.0f) *angleY += 2.0f * VV_PI;
  if (*angleZ < 0.0f) *angleZ += 2.0f * VV_PI;
}

//============================================================================
// vvVector3 Class Methods
//============================================================================

//----------------------------------------------------------------------------
/// Constructor for an empty vector
vvVector3::vvVector3()
{
  e[0] = e[1] = e[2] = 0.0;
}

//----------------------------------------------------------------------------
/// Constructor for a preset vector
vvVector3::vvVector3(float x, float y, float z)
{
  e[0] = x;
  e[1] = y;
  e[2] = z;
}

//----------------------------------------------------------------------------
/// Constructor for a copy of a vector
vvVector3::vvVector3(const vvVector3* v)
{
  e[0] = v->e[0];
  e[1] = v->e[1];
  e[2] = v->e[2];
}

//----------------------------------------------------------------------------
/** Cross product of two vectors.
 */
vvVector3 vvVector3::operator^ (const vvVector3 operand) const
{
  vvVector3 tmp;
  tmp = this;
  tmp.cross(&operand);
  return tmp;
}

//----------------------------------------------------------------------------
/// Overload subscription operator.
float &vvVector3::operator[](int index)
{
  return e[index];
}

//----------------------------------------------------------------------------
/// Overload subscription operator.
float vvVector3::operator[](int index) const
{
  return e[index];
}

//----------------------------------------------------------------------------
/// Set all elements of a vector.
void vvVector3::set(float x, float y, float z)
{
  e[0] = x;
  e[1] = y;
  e[2] = z;
}

//----------------------------------------------------------------------------
/// Get all elements of a vector.
void vvVector3::get(float* x, float* y, float* z)
{
  *x = e[0];
  *y = e[1];
  *z = e[2];
}

//----------------------------------------------------------------------------
/// Copy a vector
void vvVector3::copy(const vvVector3* v)
{
  e[0] = v->e[0];
  e[1] = v->e[1];
  e[2] = v->e[2];
}

//----------------------------------------------------------------------------
/// Copy a vector
void vvVector3::copy(const vvVector3& v)
{
  e[0] = v.e[0];
  e[1] = v.e[1];
  e[2] = v.e[2];
}

//----------------------------------------------------------------------------
/// Copy a vvVector4
void vvVector3::copy(const vvVector4* v)
{
  e[0] = v->e[0] / v->e[3];
  e[1] = v->e[1] / v->e[3];
  e[2] = v->e[2] / v->e[3];
}

//----------------------------------------------------------------------------
/// Add two vectors
void vvVector3::add(const vvVector3* v)
{
  e[0] += v->e[0];
  e[1] += v->e[1];
  e[2] += v->e[2];
}

//----------------------------------------------------------------------------
/// Add the same scalar value to each component.
void vvVector3::add(float val)
{
  e[0] += val;
  e[1] += val;
  e[2] += val;
}

//----------------------------------------------------------------------------
/// Add a different scalar value to each component.
void vvVector3::add(float x, float y, float z)
{
  e[0] += x;
  e[1] += y;
  e[2] += z;
}

//----------------------------------------------------------------------------
/// Subtract a vector from another
void vvVector3::sub(const vvVector3* v)
{
  e[0] -= v->e[0];
  e[1] -= v->e[1];
  e[2] -= v->e[2];
}

//----------------------------------------------------------------------------
/// Subtract a scalar value
void vvVector3::sub(float val)
{
  e[0] -= val;
  e[1] -= val;
  e[2] -= val;
}

//----------------------------------------------------------------------------
/// Scale a vector by a scalar
void vvVector3::scale(float s)
{
  e[0] *= s;
  e[1] *= s;
  e[2] *= s;
}

//----------------------------------------------------------------------------
/// Scale a vector by the elements of another vector
void vvVector3::scale(const vvVector3* v)
{
  e[0] *= v->e[0];
  e[1] *= v->e[1];
  e[2] *= v->e[2];
}

//----------------------------------------------------------------------------
/// Scale a vector by different scalars for each component
void vvVector3::scale(float x, float y, float z)
{
  e[0] *= x;
  e[1] *= y;
  e[2] *= z;
}

//----------------------------------------------------------------------------
/// Return the dot product of two vectors
float vvVector3::dot(const vvVector3* v) const
{
  return e[0] * v->e[0] + e[1] * v->e[1] + e[2] * v->e[2];
}

//----------------------------------------------------------------------------
/// Return the angle (in radians) between two vectors
float vvVector3::angle(vvVector3* v)
{
  float multLength, div;                          // precomputed values

  multLength = this->length() * v->length();
  if (multLength == 0.0) return 0.0;

  div = this->dot(v) / multLength;

  if (div < -1.0) return (float)(VV_PI / 2.0);
  if (div > 1.0)  return 0.0;

  return acosf(div);
}

//----------------------------------------------------------------------------
/// Create the cross product of two vectors
void vvVector3::cross(const vvVector3* v)
{
  vvVector3 bak(this);

  e[0] = bak.e[1] * v->e[2] - bak.e[2] * v->e[1];
  e[1] = bak.e[2] * v->e[0] - bak.e[0] * v->e[2];
  e[2] = bak.e[0] * v->e[1] - bak.e[1] * v->e[0];
}

//----------------------------------------------------------------------------
/// Multiplies a vector with a matrix (V' = M x V)
void vvVector3::multiply(const vvMatrix* m)
{
  int row, col;
  float v1[4], v2[4];

  for (int i=0; i<3; i++)
    v1[i] = e[i];
  v1[3] = 1.0f;

  for (row=0; row<4; ++row)
  {
    v2[row] = 0.0f;
    for(col=0; col<4; ++col)
      v2[row] += m->e[row][col] * v1[col];
  }

  for (row=0; row<3; ++row)
    e[row] = v2[row] / v2[3];
}

//----------------------------------------------------------------------------
/// Compute the distance between two points
float vvVector3::distance(const vvVector3* v)
{
  float diff[3];

  diff[0] = v->e[0] - e[0];
  diff[1] = v->e[1] - e[1];
  diff[2] = v->e[2] - e[2];
  return (float)(sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]));
}

//----------------------------------------------------------------------------
/// Compute the length of a vector
float vvVector3::length() const
{
  return (float)(sqrt((e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2])));
}

//----------------------------------------------------------------------------
/** Compute the normal of a plane which is defined by two points and a
    vector on the plane.
  @param point1,point2 two points defining the plane
  @param dir directional vector on the plane
*/
void vvVector3::planeNormalPPV(const vvVector3* point1, const vvVector3* point2,
const vvVector3* dir)
{
  vvVector3 diff;                                 // difference vector between point1 and point2

  diff.copy(point2);
  diff.sub(point1);
  this->copy(dir);
  this->cross(&diff);
}

//----------------------------------------------------------------------------
/** Computes the distance of a point from a plane.
  @param this  point to compute distance for
  @param n,p   normal and point defining a plane
  @return distance of point to plane, sign determines side of the plane on which the
          point lies: positive=side into which normal points, negative=other side
*/
float vvVector3::distPointPlane(const vvVector3* n, const vvVector3* p) const
{
  vvVector3 normal;                               // normalized plane normal
  float d;                                        // scalar component of hessian form of plane

  normal.copy(n);
  normal.normalize();
  d = -normal.dot(p);
  return (normal.dot(this) + d);
}

//----------------------------------------------------------------------------
/// Normalize a vector
void vvVector3::normalize()
{
  float d;                                        // divisor

  d = (float)(sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]));
  if (d == 0.0) return;                           // division by zero error
  e[0] /= d;
  e[1] /= d;
  e[2] /= d;
}

//----------------------------------------------------------------------------
/// Negate a vector
void vvVector3::negate()
{
  e[0] = -e[0];
  e[1] = -e[1];
  e[2] = -e[2];
}

//----------------------------------------------------------------------------
/// Compares two vectors. Returns true if equal, otherwise false
bool vvVector3::equal(const vvVector3* v)
{
  if (e[0] == v->e[0]  && e[1] == v->e[1]  && e[2] == v->e[2])
    return true;
  else return false;
}

//----------------------------------------------------------------------------
/// Creates a vector with random integer numbers in range from..to
void vvVector3::random(int from, int to)
{
  int row;

  for (row=0; row<3; ++row)
    e[row] = (float)(from + (rand() % (to - from + 1)));
}

//----------------------------------------------------------------------------
/// Creates a vector with random float numbers in range from..to
void vvVector3::random(float from, float to)
{
  int row;

  for (row=0; row<3; ++row)
    e[row] = (float)from + ((float)rand()/(float)RAND_MAX * (float)(to - from));
}

//----------------------------------------------------------------------------
/** Creates a vector with random float numbers in range from..to.
 This method is there to allow calls like random(1.0, 5.0) instead of
 random(1.0f, 5.0f)
*/
void vvVector3::random(double from, double to)
{
  random((float)from, (float)to);
}

/// Print a vector
void vvVector3::print(const char* text)
{
  cerr.setf(ios::fixed, ios::floatfield);
  cerr.precision(3);

  cerr << text << "\t" << e[0] << ", " << e[1] << ", " << e[2] << endl;
}

//----------------------------------------------------------------------------
/** Get a row vector from a matrix
  @param m    matrix
  @param row  row number [0..2]
*/
void vvVector3::getRow(const vvMatrix* m, int row)
{
  e[0] = m->e[row][0];
  e[1] = m->e[row][1];
  e[2] = m->e[row][2];
}

//----------------------------------------------------------------------------
/** Get a column vector from a matrix
  @param m    matrix
  @param col  column number [0..2]
*/
void vvVector3::getColumn(const vvMatrix* m, int col)
{
  e[0] = m->e[0][col];
  e[1] = m->e[1][col];
  e[2] = m->e[2][col];
}

//----------------------------------------------------------------------------
/// Swap two vector element matrices
void vvVector3::swap(vvVector3* v)
{
  vvVector3 bak(this);

  e[0] = v->e[0];
  e[1] = v->e[1];
  e[2] = v->e[2];
  v->e[0] = bak.e[0];
  v->e[1] = bak.e[1];
  v->e[2] = bak.e[2];
}

//----------------------------------------------------------------------------
/** Intersect a plane and a straight line.
  @param  n       normal of plane (doesn't need to be normalized)
  @param  p       an arbitrary point on the plane
  @param  v1,v2   arbitrary points on the line
  @return true if an intersection occurred, result in this pointer.
          If no intersection occurred, false is returned and all
          vector components are set to 0.0f
*/
bool vvVector3::isectPlaneLine(const vvVector3* n, const vvVector3* p,
const vvVector3* v1, const vvVector3* v2)
{
  float numer;                                    // numerator (=Zahler)
  float denom;                                    // denominator (=Nenner)
  vvVector3 diff1;                                // difference vector between v1 and v2
  vvVector3 diff2;                                // difference vector between pp and v1
  vvVector3 normal;                               // normalized normal

  normal.copy(n);
  normal.normalize();
  diff1.copy(v1);
  diff1.sub(v2);                                  // diff1 = v1 - v2
  denom = diff1.dot(&normal);                     // denom = diff1 . n
  if (denom==0.0f)                                // are ray and plane parallel?
  {
    e[0] = e[1] = e[2] = 0.0f;                    // no intersection
    return false;
  }
  diff2.copy(p);
  diff2.sub(v1);                                  // diff2 = p - v1
  numer = diff2.dot(&normal);                     // number = diff2 . n
  diff1.scale(numer / denom);                     // diff1 = diff1 * numer / denom
  this->copy(v1);
  this->add(&diff1);                              // this = v1 + diff1
  return true;
}

//----------------------------------------------------------------------------
/** Intersect a plane and a ray (line starting at a point).
  @param  n       normal of plane (must be normalized!)
  @param  p       an arbitrary point on the plane
  @param  v1      starting point of ray
  @param  v2      arbitrary point on ray
  @return true if an intersection occurred, result at this pointer.
          If no intersection occurred, false is returned and all vector
          components are set to 0.0f
*/
bool vvVector3::isectPlaneRay(const vvVector3* n, const vvVector3* p,
const vvVector3* v1, const vvVector3* v2)
{
  vvVector3 diff1;                                // difference vector between v1 and v2
  vvVector3 diff2;                                // difference vector between this and v1
  float factor;                                   // distance factor
  int i;

  // Check for intersection with straight line:
  if (this->isectPlaneLine(n, p, v1, v2) == false) return false;

  diff1.copy(v2);
  diff1.sub(v1);                                  // diff1 = v2 - v1

  diff2.copy(this);
  diff2.sub(v1);                                  // diff2 = this - v1

  // Find out how to represent diff2 by diff1 times a factor:
  factor = 0.0f;
  for (i=0; i<3; ++i)
  {
    if (diff1.e[i] != 0.0f)
      factor = diff2.e[i] / diff1.e[i];
  }

  if (factor < 0)                                 // intersection in opposite direction than ray?
  {
    this->zero();
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------
/** Compute all intersections of a plane and a cuboid.<P>
  Example call:<BR><TT>
  vvVector3 isect[6];<BR>
  isect[0].isectPlaneCuboid(...);</TT><P>
  @param  normal    normal of plane (must be normalized!)
  @param  point     an arbitrary point on the plane
  @param  minv,maxv Minimum and maximum vertices of cuboid.<BR>
                    All minimum vertex's coordinates must be smaller than
                    the maximum vertex's coordinates.<BR>
                    The cuboid's edges are assumed to be parallel to the coordinate axes.
  @return number of intersection points. The intersection points are returned
in an array, of which the first element must be the caller. There must
be <I>at least</I> 6 elements defined in the calling vector!<P>
If less than six intersections occurred, the remaining vector
components of this[] are undefined.
*/
int vvVector3::isectPlaneCuboid(const vvVector3* normal, const vvVector3* point,
const vvVector3* minv, const vvVector3* maxv/*, vvVector3* isect*/)
{
  vvVector3 p[2];                                 // two cuboid vertices defining a cuboid edge
  vvVector3* corner[2];                           // cuboid corners (copied from parameters)
  int key[4][3] =                                 // cuboid edge components
  {
    {
      0, 0, 0
    }
    ,
    {
      1, 0, 1
    }
    ,
    {
      1, 1, 0
    }
    ,
    {
      0, 1, 1
    }
  };
  int isectCnt = 0;                               // intersection counter
  int i,j;

  // Copy parameters for easier access:
  corner[0] = (vvVector3*)minv;
  corner[1] = (vvVector3*)maxv;

  // Search for intersections between texture plane (defined by texPoint and
  // normal) and texture object (0..1):
  for (i=0; i<4 && isectCnt<6; ++i)               // loop thru volume key vertices
  {
    for (j=0; j<3 && isectCnt<6; ++j)             // loop thru secondary vertices
    {
      // Compute vertices of one cuboid edge:
      p[0].set(corner[key[i][0]]->e[0], corner[key[i][1]]->e[1], corner[key[i][2]]->e[2]);
      p[1].set((j==0) ? (corner[1-key[i][0]]->e[0]) : corner[key[i][0]]->e[0],
        (j==1) ? (corner[1-key[i][1]]->e[1]) : corner[key[i][1]]->e[1],
        (j==2) ? (corner[1-key[i][2]]->e[2]) : corner[key[i][2]]->e[2]);

      // Compute intersections of one cuboid edge with the plane:
      if (this[isectCnt].isectPlaneLine(normal, point, &p[0], &p[1]))
      {
        if (this[isectCnt].e[0] >= minv->e[0] && this[isectCnt].e[0] <= maxv->e[0] &&
          this[isectCnt].e[1] >= minv->e[1] && this[isectCnt].e[1] <= maxv->e[1] &&
          this[isectCnt].e[2] >= minv->e[2] && this[isectCnt].e[2] <= maxv->e[2])
        {
          ++isectCnt;                             // accept this entry
        }
      }
    }
  }
  assert(isectCnt<7);                             // there can be at most 6 intersections
  return isectCnt;
}

//----------------------------------------------------------------------------
/** Intersect a cylinder (infinitely long) with a ray (a line
  that emerges from a point and is infinitely long in the other direction).

  @param cylBase      arbitrary point on axis of symmetry for cylinder
  @param cylAxis      direction into which cylinder extends (does not need to be normalized)
  @param cylRadius    radius of cylinder
  @param rayBase      starting point of ray
  @param rayDir       direction into which ray extends (does not need to be normalized)
  @return The return value is the number of intersection points (0..2).
          If this value is greater than 0, the closest intersection point to
          the ray starting point is returned in the vector on which the function
was executed.
*/
int vvVector3::isectRayCylinder(const vvVector3* cylBase, const vvVector3* cylAxis,
float cylRadius, const vvVector3* rayBase, const vvVector3* rayDir)
{
  vvVector3 ortho;                                // vector orthogonal to cylAxis and rayDir
  vvVector3 diff;                                 // difference vector between base points
  vvVector3 temp;                                 // temporary vector
  vvVector3 closest;                              // closest point on ray-line to cylinder axis
  vvVector3 cylAxisN;                             // normalized cylinder axis
  vvVector3 rayDirN;                              // normalized ray direction vector
  float     dist;                                 // distance between cylinder axis and closest point on ray
  float     t;                                    // distance of closest from rayBase
  float     s;                                    // distance of intersections from closest
  float     len;                                  // length of ortho before normalization
  float     i1, i2;                               // distances of intersections from rayBase
  float     first;                                // first intersection = intersection closest to rayBase
  int       i;                                    // counter

  // Compute normalized direction vectors:
  cylAxisN.copy(cylAxis);
  cylAxisN.normalize();
  rayDirN.copy(rayDir);
  rayDirN.normalize();

  // Compute distance between closest point on line and cylinder axis:
  ortho.copy(rayDir);
  ortho.cross(cylAxis);
  len = ortho.length();
  ortho.normalize();
  if (ortho.isZero())                             // is the ray parallel to the cylinder?
  {
    this->zero();
    return 0;
  }
  diff.copy(rayBase);
  diff.sub(cylBase);
  dist = (float)fabs(diff.dot(&ortho));
  if (dist > cylRadius)                           // do the ray and the cylinder intersect at all?
  {
    this->zero();
    return 0;
  }

  // Find point on line closest to cylinder axis:
  temp.copy(&diff);
  temp.cross(&cylAxisN);
  t = -1.0f * temp.dot(&ortho) / len;
  closest.copy(&rayDirN);
  closest.scale(t);
  closest.add(rayBase);

  // Find intersections of ray-line and cylinder:
  s = sqrtf(cylRadius * cylRadius - dist * dist);
  i1 = t + s;                                     // closest intersection
  i2 = t - s;                                     // second intersection

  // Only the positive values of i1 and i2 are located on the ray:
  if (i1<0.0f && i2<0.0f)
  {
    this->zero();
    return 0;
  }
  if (i1 < 0.0f) first = i2;
  else if (i2 < 0.0f) first = i1;
  else first = (i1 < i2) ? i1 : i2;
  for (i=0; i<3; ++i)
    this->e[i] = rayBase->e[i] + first * rayDirN.e[i];
  if (i1<0.0f || i2<0.0f) return 1;
  else return 2;
}

//----------------------------------------------------------------------------
/** Intersect a ray with a triangle.
  @param rayPt  starting point of ray
  @param rayDir direction of ray
  @param tri1   first triangle vertex
  @param tri2   second triangle vertex
  @param tri3   third triangle vertex
  @return true if ray intersects triangle, this pointer contains intersection point.
          false if no intersection, this pointer contains zero vector.
*/
bool vvVector3::isectRayTriangle(const vvVector3* rayPt, const vvVector3* rayDir,
const vvVector3* tri1, const vvVector3* tri2, const vvVector3* tri3)
{
  vvVector3 rayPt2;                               // arbitrary point on ray
  vvVector3 normal;                               // normal vector of triangle (normalized)
  vvVector3 diff1;                                // vector from tri1 to tri2
  vvVector3 diff2;                                // vector from tri1 to tri3
  vvVector3 sideNormal;                           // normal vector made from one triangle side and the triangle normal

  // Compute second point on ray:
  rayPt2.copy(rayPt);
  rayPt2.add(rayDir);

  // Compute triangle normal:
  diff1.copy(tri2);
  diff1.sub(tri1);
  diff2.copy(tri3);
  diff2.sub(tri1);
  normal.copy(&diff1);
  normal.cross(&diff2);
  normal.normalize();

  // Compute intersection of ray and triangle plane:
  if (this->isectPlaneRay(&normal, tri1, rayPt, &rayPt2) == false) return false;

  // Compute three triangle side normals and check if intersection point lies on same
  // side as third triangle point, respectively:
  sideNormal.planeNormalPPV(tri1, tri2, &normal);
  if (vvVecmath::sgn(tri3->distPointPlane(&sideNormal, tri1)) ==
    vvVecmath::sgn(this->distPointPlane(&sideNormal, tri1)))
  {
    sideNormal.planeNormalPPV(tri1, tri3, &normal);
    if (vvVecmath::sgn(tri2->distPointPlane(&sideNormal, tri1)) ==
      vvVecmath::sgn(this->distPointPlane(&sideNormal, tri1)))
    {
      sideNormal.planeNormalPPV(tri2, tri3, &normal);
      if (vvVecmath::sgn(tri1->distPointPlane(&sideNormal, tri2)) ==
        vvVecmath::sgn(this->distPointPlane(&sideNormal, tri2)))
      {
        return true;
      }
    }
  }
  this->zero();
  return false;
}

//----------------------------------------------------------------------------
/** Intersect two lines.
  Source: http://astronomy.swin.edu.au/~pbourke/geometry/lineline3d/
   Calculate the line segment PaPb that is the shortest route between
   two lines P1P2 and P3P4. Calculate also the values of mua and mub where
      Pa = P1 + mua (P2 - P1)
      Pb = P3 + mub (P4 - P3)
   Return FALSE if no solution exists.
  @param  p1,p2  points on each line
  @param  v1,v2  each line's direction vectors
  @return shortest distance between lines (0.0f if lines intersect).
          If lines don't intersect, the this vector will be the point
on line 1 that is closest to line 2.
*/
float vvVector3::isectLineLine(const vvVector3& pt1, const vvVector3& v1,
const vvVector3& pt2, const vvVector3& v2)
{
  const float EPS = 0.0001f;
  vvVector3 p1, p2, p3, p4, p13, p43, p21, pa, pb;
  float d1343, d4321, d1321, d4343, d2121;
  float numer, denom, mua, mub;

  /*
    float dist;

    // Create plane from line 1 and direction of line 2:
    vvPlane plane(p1, v1, v2);

    // Compute shortest distance between lines:
    this->copy(&p2);
    dist = this->distPointPlane(&plane.normal, &plane.point);

    return dist;
  */
  p1.copy(&pt1);
  p2.copy(&pt1);
  p2.add(&v1);
  p3.copy(&pt2);
  p4.copy(&pt2);
  p4.add(&v2);

  p13.e[0] = p1.e[0] - p3.e[0];
  p13.e[1] = p1.e[1] - p3.e[1];
  p13.e[2] = p1.e[2] - p3.e[2];
  p43.e[0] = p4.e[0] - p3.e[0];
  p43.e[1] = p4.e[1] - p3.e[1];
  p43.e[2] = p4.e[2] - p3.e[2];
  if (fabs(p43.e[0]) < EPS && fabs(p43.e[1])  < EPS && fabs(p43.e[2])  < EPS)
    return(0.0f);
  p21.e[0] = p2.e[0] - p1.e[0];
  p21.e[1] = p2.e[1] - p1.e[1];
  p21.e[2] = p2.e[2] - p1.e[2];
  if (fabs(p21.e[0])  < EPS && fabs(p21.e[1])  < EPS && fabs(p21.e[2])  < EPS)
    return(0.0f);

  d1343 = p13.e[0] * p43.e[0] + p13.e[1] * p43.e[1] + p13.e[2] * p43.e[2];
  d4321 = p43.e[0] * p21.e[0] + p43.e[1] * p21.e[1] + p43.e[2] * p21.e[2];
  d1321 = p13.e[0] * p21.e[0] + p13.e[1] * p21.e[1] + p13.e[2] * p21.e[2];
  d4343 = p43.e[0] * p43.e[0] + p43.e[1] * p43.e[1] + p43.e[2] * p43.e[2];
  d2121 = p21.e[0] * p21.e[0] + p21.e[1] * p21.e[1] + p21.e[2] * p21.e[2];

  denom = d2121 * d4343 - d4321 * d4321;
  if (fabs(denom) < EPS) return(0.0f);
  numer = d1343 * d4321 - d1321 * d4343;

  mua = numer / denom;
  mub = (d1343 + d4321 * (mua)) / d4343;

  pa.e[0] = p1.e[0] + mua * p21.e[0];
  pa.e[1] = p1.e[1] + mua * p21.e[1];
  pa.e[2] = p1.e[2] + mua * p21.e[2];
  pb.e[0] = p3.e[0] + mub * p43.e[0];
  pb.e[1] = p3.e[1] + mub * p43.e[1];
  pb.e[2] = p3.e[2] + mub * p43.e[2];

  this->copy(&pa);

  return(pa.distance(&pb));
}

//----------------------------------------------------------------------------
/** Check if two points are on the same side relative to a line formed by
  two points. The third coordinate of all points must be zero, as this
  is a 2D algorithm!
  Algorithm:<p>
  cp1 = CrossProduct(b-a, p1-a)<br>
  cp2 = CrossProduct(b-a, p2-a)<br>
  if DotProduct(cp1, cp2) >= 0 then return true else return false<p>
  @param p1,p2 points to check
  @param a,b points forming the line
*/
bool vvVector3::isSameSideLine2D(const vvVector3* p1, const vvVector3* p2,
const vvVector3* a, const vvVector3* b)
{
  vvVector3 diff1, diff2;
  vvVector3 cp1, cp2;
  vvVector3 dp;

  assert(p1->e[2]==0.0f && p2->e[2]==0.0f && a->e[2]==0.0f && b->e[2]==0.0f);
  diff1.e[0] = p1->e[0] - a->e[0];
  diff1.e[1] = p1->e[1] - a->e[1];
  diff2.e[0] = p2->e[0] - a->e[0];
  diff2.e[1] = p2->e[1] - a->e[1];
  cp1.e[0] = b->e[0] - a->e[0];
  cp1.e[1] = b->e[1] - a->e[1];
  cp2.copy(&cp1);
  cp1.cross(&diff1);
  cp2.cross(&diff2);
  dp.copy(&cp1);
  if (dp.dot(&cp2) >= 0.0f) return true;
  else return false;
}

//----------------------------------------------------------------------------
/** Check if a point is in a triangle (so far third coordinate must be zero!)
  @param this point to check
  @param v1,v2,v3 triangle vertices
*/
bool vvVector3::isInTriangle(const vvVector3* v1, const vvVector3* v2, const vvVector3* v3)
{
  assert(this->e[2]==0.0f && v1->e[2]==0.0f && v2->e[2]==0.0f && v3->e[2]==0.0f);

  if (isSameSideLine2D(this, v1, v2, v3) && isSameSideLine2D(this, v2, v1, v3) &&
    isSameSideLine2D(this, v3, v1, v2)) return true;
  else return false;
}

//----------------------------------------------------------------------------
/** Cyclically sort a number of vectors in an array starting with this vector.
  Example call:<BR><TT>
  vvVector3 vecArray[6];<BR>
  vecArray[0].cyclicSort(6);</TT><P>
  @param numVectors number of vectors in 'this' array to be sorted
  @param axis     normal vector defining cycle direction
*/
void vvVector3::cyclicSort(int numVectors, const vvVector3* axis)
{
  vvVector3* diff;                                // difference vectors between pairs of points
  vvVector3 normal;                               // normal vector
  int i,j;                                        // loop counters
  bool swapped;                                   // true = vectors were swapped

  // Allocate array of difference vectors:
  diff = new vvVector3[numVectors-1];

  // Compute difference vectors:
  for (i=0; i<numVectors-1; ++i)
  {
    diff[i].copy(&this[i+1]);
    diff[i].sub(&this[0]);
  }

  // Sort vectors:
  swapped = true;
  while (swapped)
  {
    swapped = false;
    for (i=0; i<numVectors-2 && swapped==false; ++i)
    {
      for (j=i+1; j<numVectors-1 && swapped==false; ++j)
      {
        normal.copy(&diff[i]);
        normal.cross(&diff[i+1]);
        normal.normalize();
        if (normal.dot(axis) < 0.0f)              // do normals point into opposite directions?
        {
          this[i+1].swap(&this[j+1]);             // swap points
          diff[i].swap(&diff[j]);                 // swap difference vectors
          swapped = true;
        }
      }
    }
  }
  delete[] diff;
}

//----------------------------------------------------------------------------
/// Set all components to zero.
void vvVector3::zero()
{
  e[0] = e[1] = e[2] = 0.0f;
}

//----------------------------------------------------------------------------
/// @return true if all vector elements are zero
bool vvVector3::isZero()
{
  if (e[0]==0.0f && e[1]==0.0f && e[2]==0.0f) return true;
  else return false;
}

//----------------------------------------------------------------------------
/** Convert cartesian coordinates to spherical coordinates
  The sphere coordinate equations are taken from Bronstein page 154f.<BR>
  The used coordinate system is:<PRE>
      y
      |
      |___x
     /
    z
  </PRE>

  @param p      point to convert
@param r      distance from given point to center of sphere
@param phi    angle in the x/z axis, starting at (0|0|1), passing
(1|0|0), (0|0|-1), and (-1|0|0) in this order.
@param theta  angle from the x/y/z position vector to the vector (0|1|0).
@return spherical coordinates in r, phi, and theta
*/
void vvVector3::getSpherical(float* r, float* phi, float* theta)
{
  vvVector3 upvec;                                // up vector (points into negative y direction [voxel space]

  // Compute sphere coordinates of current destination voxel:
  upvec.set(0.0f, 1.0f, 0.0f);
  *r = length();
  *theta = angle(&upvec);

  // Phi must be computed differently for each quadrant:
  if (e[0] >= 0.0f)
  {
    if (e[2] >= 0.0f)
      *phi = (float)atan(e[0] / e[2]);            // first quadrant
    else
    {
      *phi = (float)atan(e[0] / -e[2]);           // second quadrant
      *phi = VV_PI - *phi;
    }
  }
  else
  {
    if (e[2] <= 0.0f)
    {
      *phi = (float)atan(e[0] / e[2]);            // third quadrant
      *phi += VV_PI;
    }
    else
    {
      *phi = (float)atan(-e[0] / e[2]);           // fourth quadrant
      *phi = 2.0f * VV_PI - *phi;
    }
  }
}

/** Compute the direction cosines of a vector.
  @param src vector from which to compute the direction cosines
  @return direction cosines in 'this' vector
*/
void vvVector3::directionCosines(const vvVector3* src)
{
  for (int i=0; i<3; ++i)
    e[i] = cosf(atanf(src->e[i]));
}

//
// Operators...
//
/** Compute the sum of two vectors in formula notation.
Example call:<BR><TT>
  vvVector3 vecA(0,0,0), vecB(0,1,0), vecSum;<BR>
  vecSum = vecA + vecB; </TT><P>
*/
vvVector3 vvVector3::operator + ( const vvVector3 &other )
{
  vvVector3 vResult(0.0f, 0.0f, 0.0f);

  vResult.e[0] = e[0] + other.e[0];
  vResult.e[1] = e[1] + other.e[1];
  vResult.e[2] = e[2] + other.e[2];

  return vResult;
}

vvVector3 vvVector3::operator + ( void ) const
{
  return *this;
}

/** Compute the difference of two vectors in formula notation.
Example call:<BR><TT>
  vvVector3 vecA(1,1,0), vecB(0,1,0), vecDiff;<BR>
  vecDiff = vecA - vecB; </TT><P>
*/
vvVector3 vvVector3::operator - ( const vvVector3 &other )
{
  vvVector3 vResult(0.0f, 0.0f, 0.0f);

  vResult.e[0] = e[0] - other.e[0];
  vResult.e[1] = e[1] - other.e[1];
  vResult.e[2] = e[2] - other.e[2];

  return vResult;
}

vvVector3 vvVector3::operator - ( void ) const
{
  vvVector3 vResult(-e[0], -e[1], -e[2]);

  return vResult;
}

vvVector3 vvVector3::operator * ( const vvVector3 &other )
{
  vvVector3 vResult(0.0f, 0.0f, 0.0f);

  vResult.e[0] = e[0] * other.e[0];
  vResult.e[1] = e[1] * other.e[1];
  vResult.e[2] = e[2] * other.e[2];

  return vResult;
}

vvVector3 vvVector3::operator * ( const float scalar )
{
  vvVector3 vResult(0.0f, 0.0f, 0.0f);

  vResult.e[0] = e[0] * scalar;
  vResult.e[1] = e[1] * scalar;
  vResult.e[2] = e[2] * scalar;

  return vResult;
}

vvVector3 operator * ( const float scalar, const vvVector3 &other )
{
  vvVector3 vResult(0.0f, 0.0f, 0.0f);

  vResult.e[0] = other.e[0] * scalar;
  vResult.e[1] = other.e[1] * scalar;
  vResult.e[2] = other.e[2] * scalar;

  return vResult;
}

vvVector3& vvVector3::operator = ( const vvVector3 &other )
{
  e[0] = other.e[0];
  e[1] = other.e[1];
  e[2] = other.e[2];

  return *this;
}

vvVector3& vvVector3::operator += ( const vvVector3 &other )
{
  e[0] += other.e[0];
  e[1] += other.e[1];
  e[2] += other.e[2];

  return *this;
}

vvVector3& vvVector3::operator -= ( const vvVector3 &other )
{
  e[0] -= other.e[0];
  e[1] -= other.e[1];
  e[2] -= other.e[2];

  return *this;
}

//============================================================================
// vvVector4 Class Methods
//============================================================================

//----------------------------------------------------------------------------
/// Copy a vector
vvVector4::vvVector4()
{
  e[0] = e[1] = e[2] = e[3] = 0.0f;
}

//----------------------------------------------------------------------------
/// Constructor for a copy of a vector
vvVector4::vvVector4(const vvVector4* v)
{
  e[0] = v->e[0];
  e[1] = v->e[1];
  e[2] = v->e[2];
  e[3] = v->e[3];
}

//----------------------------------------------------------------------------
/// Overload subscription operator.
float &vvVector4::operator[](int index)
{
  return e[index];
}

//----------------------------------------------------------------------------
/// Overload subscription operator.
float vvVector4::operator[](int index) const
{
  return e[index];
}

//----------------------------------------------------------------------------
/// Set elements of a 4 vector
void vvVector4::set(float a, float b, float c, float d)
{
  e[0] = a;
  e[1] = b;
  e[2] = c;
  e[3] = d;
}

//----------------------------------------------------------------------------
/// Multiplies a vector with a matrix (V' = M x V)
void vvVector4::multiply(const vvMatrix* m)
{
  int row, col;
  vvVector4 bak(this);

  for (row=0; row<4; ++row)
  {
    e[row] = 0.0f;
    for(col=0; col<4; ++col)
      e[row] += m->e[row][col] * bak.e[col];
  }
}

//----------------------------------------------------------------------------
/// Copy a vector
void vvVector4::copy(const vvVector4* v)
{
  e[0] = v->e[0];
  e[1] = v->e[1];
  e[2] = v->e[2];
  e[3] = v->e[3];
}

//----------------------------------------------------------------------------
/// Print a 4-vector
void vvVector4::print(const char* text)
{
  cerr.setf(ios::fixed, ios::floatfield);
  cerr.precision(3);

  cerr << text << "\t" << e[0] << ", " << e[1] << ", " << e[2] << ", " << e[3] << endl;
  cerr << "   Normalized: " << e[0]/e[3] << ", " << e[1]/e[3] << ", " <<
    e[2]/e[3] << endl;
}

//----------------------------------------------------------------------------
/// Constructor for point-normal format.
vvPlane::vvPlane(const vvVector3& p, const vvVector3& n)
{
  _point.copy(&p);
  _normal.copy(&n);
  _normal.normalize();
}

//----------------------------------------------------------------------------
/// Constructor for point-vector-vector format.
vvPlane::vvPlane(const vvVector3& p, const vvVector3& dir1, const vvVector3& dir2)
{
  _point.copy(&p);
  _normal.copy(&dir1);
  _normal.cross(&dir2);
  _normal.normalize();
}

//----------------------------------------------------------------------------
/** Check if two points are on the same side of the plane.
  @param p1,p2 points to check
  @return true if points are on the same side
*/
bool vvPlane::isSameSide(const vvVector3& p1, const vvVector3& p2) const
{
  if (vvVecmath::sgn(dist(p1)) == vvVecmath::sgn(dist(p2))) return true;
  else return false;
}

//----------------------------------------------------------------------------
/** Computes the distance of a point from the plane.
  @param p   normal and point defining a plane
  @return distance of point to plane, sign determines side of the plane on which the
          point lies: positive=side into which normal points, negative=other side
*/
float vvPlane::dist(const vvVector3& p) const
{
  float d;                                        // scalar component of hessian form of plane

  d = -_normal.dot(&_point);
  return (_normal.dot(&p) + d);
}

//============================================================================
// Functions for STANDALONE mode
//============================================================================

#ifdef VV_STANDALONE

//----------------------------------------------------------------------------
/// test routines for vvMatrix
void testMatrix()
{
  vvMatrix m1;
  vvMatrix m2;
  vvMatrix m3;
  float glmatrix[16] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
  float left, right, bottom, top, nearPlane, farPlane;

  m1.identity();
  m1.print("m1 =");
  m2.identity();
  m2.scale(3.0f);
  m2.print("m2 =");
  m2.translate(2.0, 3.0, 4.0);
  m2.print("m2.translate(2.0, 3.0, 4.0)=");
  m3.e[0][0] = 1.0;
  m3.e[0][2] = 2.0;
  m3.print("m3 =");
  m3.rotate((float)VV_PI / 2.0f, 1.0f, 0.0f, 0.0f);
  m3.print("m3.rotate(M_PI / 2.0, 1.0, 0.0, 0.0)=");
  m1.random(0, 5);
  m1.print("m1.random(0, 5)=");
  m2.random(0, 5);
  m2.print("m2.random(0, 5)=");
  m1.multiplyPre(&m2);
  m1.print("m1.multiply(&m2)");
  m2.random(1.0f, 10.0f);
  m2.print("m2.random(1.0, 10.0)=");
  m1.makeGL((float*)m2.e);
  m2.print("m1.makeGL((float*)m2.e)=");
  m2.getGL(glmatrix);
  m2.print("m2.getGL(glmatrix)=");
  m2.transpose();
  m2.print("m2.transpose()=");

  m1.random(1, 9);
  m1.print("m1.random(1, 9) =");
  m1.rotOnly();
  m1.print("m1.rotOnly()= ");

  m1.random(1, 9);
  m1.print("m1.random(1, 9) =");
  m1.transOnly();
  m1.print("m1.transOnly()= ");

  m1.random(1, 9);
  m1.print("m1.random(1, 9) =");
  m1.invert();
  m1.print("m1.invert()=");

  m1.setProjOrtho(-2.0, 5.0, -4.0, 7.0, -250.0, 999.0);
  m1.print("m1.setProjOrtho(-2.0, 5.0, -4.0, 7.0, -250.0, 999.0) =");
  m1.getProjOrtho(&left, &right, &bottom, &top, &nearPlane, &farPlane);
  cerr << "left=" << left << ", right=" << right << ", bottom=" << bottom <<
    ", top=" << top << ", nearPlane=" << nearPlane << ", farPlane=" << farPlane << endl;

  m1.setProjPersp(-12.0, 15.0, -14.0, 17.0, -25.0, 99.0);
  m1.print("m1.setProjPersp(-12.0, 15.0, -14.0, 17.0, -25.0, 99.0) =");
  m1.getProjPersp(&left, &right, &bottom, &top, &nearPlane, &farPlane);
  cerr << "left=" << left << ", right=" << right << ", bottom=" << bottom <<
    ", top=" << top << ", nearPlane=" << nearPlane << ", farPlane=" << farPlane << endl;
}

//----------------------------------------------------------------------------
/// test routines for vvVector3 and vvVector4
void testVector()
{
  vvMatrix m;
  vvVector3 v1(1.0, 2.0, 3.0);
  vvVector3 v2, v3, v4, v5, v6;
  float result;

  v3.print("v3=");
  v4.random(0.0, 9.0);
  v4.print("v4.random(0.0, 0.9)= ");
  v1.print("v1=");
  v2.random(1, 5);
  v2.print("v2.random(1, 5)= ");

  v3.copy(&v2);
  v3.add(&v1);
  v3.print("v2.add(&v1)= ");

  v3.copy(&v2);
  v3.sub(&v1);
  v3.print("v2.sub(&v1)= ");

  v3.copy(&v2);
  result = v3.dot(&v1);
  cerr << "v2.dot(&v1)= " << result << endl;

  v3.copy(&v2);
  v3.cross(&v1);
  v3.print("v2.cross(&v1)= ");

  v3.copy(&v2);
  result = v3.distance(&v1);
  cerr << "v2.distance(&v1)= " << result << endl;

  v3.copy(&v2);
  v3.scale(3.0);
  v3.print("v2.scale(3.0)= ");

  v3.copy(&v2);
  v3.normalize();
  v3.print("v2.normalize= ");

  m.random(1, 9);
  m.print("Matrix m=");
  v3.copy(&v2);
  v3.multiply(&m);
  v3.print("v2.multiply(&m)= ");

  v2.set(2.0, 3.0, 0.0);
  v2.print("v2= ");
  v3.set(5.0, 0.0, 0.0);
  v3.print("v3= ");
  cerr << "v3.angle(&v2)= " << v3.angle(&v2) * 180.0 / VV_PI << endl;

  v3.set(1.0, 2.0, 3.0);
  v3.print("v3=");
  v4.set(-6.0, -2.0, 9.0);
  v4.print("v4=");
  v3.swap(&v4);
  cerr << "v3.swap(&v4)" << endl;
  v3.print("v3=");
  v4.print("v4=");

  int numIntersections;
  v2.set(0.0, 0.0, 0.0);
  v3.set(0.0, 0.0, 1.0);
  v4.set(-2.0, -3.0, 0.0);
  v5.set(0.0, 1.0, 0.0);
  numIntersections = v1.isectRayCylinder(&v2, &v3, 3.0f, &v4, &v5);

  v1.set(1.0, 1.0, 1.0);
  v1.print("v1=");
  v2.set(1.0, 1.0, 1.0);
  v2.print("v2=");
  v3.set(0.0, 0.0, 1.0);
  v3.print("v3=");
  v4.set(1.0, 0.0, 1.0);
  v4.print("v4=");
  v3.isectPlaneRay(&v1, &v2, &v4, &v3);
  v3.print("isectPlaneRay(&v1, &v2, &v4, &v3)=");

  v1.set(1.0, 1.0, 1.0);
  v2.set(4.0, 1.0, 1.0);
  v3.set(4.0, 3.0, 1.0);
  v4.set(0.9f, 1.0, 0.0);
  v5.set(0.0, 0.0, 1.0);
  v6.isectRayTriangle(&v4, &v5, &v1, &v2, &v3);
  v6.print("isectRayTriangle");

  v1.set(0.0, 0.0, 0.0);
  v2.set(0.0, 0.0, 1.0);
  v3.set(1.0, 2.0, 0.0);
  v4.set(0.0, 0.0, 1.0);
  cerr << "dist=" << v5.isectLineLine(v1, v2, v3, v4) << endl;
  v5.print("isectLineLine");
}

//----------------------------------------------------------------------------
/// main routine, only used in STANDALONE mode
int main(int, char*)
{
  srand(42);
  testMatrix();
  testVector();
  return 1;
}
#endif

/////////////////
// End of File
/////////////////
