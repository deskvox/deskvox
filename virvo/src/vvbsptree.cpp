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

#include <cmath>
#include <set>

#include "vvbrick.h"
#include "vvbsptree.h"
#include "vvgltools.h"
#include "vvopengl.h"
#include "vvtoolshed.h"

//============================================================================
// vvAABB Method Definitions
//============================================================================

vvAABB::vvAABB(const vvVector3& bottomLeftBackCorner,
               const vvVector3& topRightFrontCorner)
{
  _bottomLeftBackCorner = bottomLeftBackCorner;
  _topRightFrontCorner = topRightFrontCorner;
}

float vvAABB::calcWidth() const
{
  return calcMaxExtent(vvVector3(1, 0, 0)) - calcMinExtent(vvVector3(1, 0, 0));
}

float vvAABB::calcHeight() const
{
  return calcMaxExtent(vvVector3(0, 1, 0)) - calcMinExtent(vvVector3(0, 1, 0));
}

float vvAABB::calcDepth() const
{
  return calcMaxExtent(vvVector3(0, 0, 1)) - calcMinExtent(vvVector3(0, 0, 1));
}

const vvBoxCorners &vvAABB::calcVertices()
{
  for(int i=0; i<8; ++i)
  {
    // return the vertices in the necessary order
    int d=i;
    if(i>=2 && i<=5)
      d ^= 1;

    for(int c=0; c<3; ++c)
    {
      _vertices[i].e[c] = (1<<c)&d ? _bottomLeftBackCorner.e[c] : _topRightFrontCorner.e[c];
    }
  }

  return _vertices;
}

float vvAABB::calcMinExtent(const vvVector3& axis) const
{
  return _bottomLeftBackCorner.e[0] * axis.e[0]
       + _bottomLeftBackCorner.e[1] * axis.e[1]
       + _bottomLeftBackCorner.e[2] * axis.e[2];
}

float vvAABB::calcMaxExtent(const vvVector3& axis) const
{
  return _topRightFrontCorner.e[0] * axis.e[0]
       + _topRightFrontCorner.e[1] * axis.e[1]
       + _topRightFrontCorner.e[2] * axis.e[2];
}

vvVector3 vvAABB::calcCenter() const
{
  return vvVector3((_bottomLeftBackCorner.e[0] + _topRightFrontCorner.e[0]) * 0.5f,
                   (_bottomLeftBackCorner.e[1] + _topRightFrontCorner.e[1]) * 0.5f,
                   (_bottomLeftBackCorner.e[2] + _topRightFrontCorner.e[2]) * 0.5f);
}

vvRect* vvAABB::getProjectedScreenRect()
{
  GLdouble modelview[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);

  GLdouble projection[16];
  glGetDoublev(GL_PROJECTION_MATRIX, projection);

  const vvGLTools::Viewport viewport = vvGLTools::getViewport();

  calcVertices();
  float minX = FLT_MAX;
  float minY = FLT_MAX;
  float maxX = -FLT_MAX;
  float maxY = -FLT_MAX;

  for (int i = 0; i < 8; ++i)
  {
    GLdouble vertex[3];
    gluProject(_vertices[i][0], _vertices[i][1], _vertices[i][2],
               modelview, projection, viewport.values,
               &vertex[0], &vertex[1], &vertex[2]);

    if (vertex[0] < minX)
    {
      minX = static_cast<float>(vertex[0]);
    }
    if (vertex[0] > maxX)
    {
      maxX = static_cast<float>(vertex[0]);
    }

    if (vertex[1] < minY)
    {
      minY = static_cast<float>(vertex[1]);
    }
    if (vertex[1] > maxY)
    {
      maxY = static_cast<float>(vertex[1]);
    }
  }

  vvRect* result = new vvRect;
  result->x = max(0, static_cast<int>(floorf(minX)));
  result->y = max(0, static_cast<int>(floorf(minY)));
  result->width = min(static_cast<int>(ceilf(fabsf(maxX - minX))), viewport[2]);
  result->height = min(static_cast<int>(ceilf(fabsf(maxY - minY))), viewport[3]);

  return result;
}

void vvAABB::render()
{
  const vvVector3 (&vertices)[8] = calcVertices();
  glDisable(GL_LIGHTING);
  glBegin(GL_LINES);
    glColor3f(1.0f, 1.0f, 1.0f);

    glVertex3f(vertices[0].e[0], vertices[0].e[1], vertices[0].e[2]);
    glVertex3f(vertices[1].e[0], vertices[1].e[1], vertices[1].e[2]);

    glVertex3f(vertices[1].e[0], vertices[1].e[1], vertices[1].e[2]);
    glVertex3f(vertices[2].e[0], vertices[2].e[1], vertices[2].e[2]);

    glVertex3f(vertices[2].e[0], vertices[2].e[1], vertices[2].e[2]);
    glVertex3f(vertices[3].e[0], vertices[3].e[1], vertices[3].e[2]);

    glVertex3f(vertices[3].e[0], vertices[3].e[1], vertices[3].e[2]);
    glVertex3f(vertices[0].e[0], vertices[0].e[1], vertices[0].e[2]);


    glVertex3f(vertices[4].e[0], vertices[4].e[1], vertices[4].e[2]);
    glVertex3f(vertices[5].e[0], vertices[5].e[1], vertices[5].e[2]);

    glVertex3f(vertices[5].e[0], vertices[5].e[1], vertices[5].e[2]);
    glVertex3f(vertices[6].e[0], vertices[6].e[1], vertices[6].e[2]);

    glVertex3f(vertices[6].e[0], vertices[6].e[1], vertices[6].e[2]);
    glVertex3f(vertices[7].e[0], vertices[7].e[1], vertices[7].e[2]);

    glVertex3f(vertices[7].e[0], vertices[7].e[1], vertices[7].e[2]);
    glVertex3f(vertices[4].e[0], vertices[4].e[1], vertices[4].e[2]);
  glEnd();
  glEnable(GL_LIGHTING);
}

void vvAABB::print()
{
  _bottomLeftBackCorner.print();
  _topRightFrontCorner.print();
}

//============================================================================
// vvHalfSpace Method Definitions
//============================================================================

vvHalfSpace::vvHalfSpace()
{
  _firstSon = NULL;
  _nextBrother = NULL;

  _splitPlane = NULL;
  _bricks = NULL;
  _boundingBox = NULL;
  _projectedScreenRect = NULL;
}

vvHalfSpace::~vvHalfSpace()
{
  delete _splitPlane;
  delete _firstSon;
  delete _nextBrother;
  delete _boundingBox;
  delete _projectedScreenRect;
}

void vvHalfSpace::accept(vvVisitor* visitor)
{
  visitor->visit(this);
}

void vvHalfSpace::addChild(vvHalfSpace* child)
{
  if (_firstSon == NULL)
  {
    // First son.
    _firstSon = child;
  }
  else
  {
    // Second son is _firstSon's next brother... .
    _firstSon->_nextBrother = child;
  }
}

bool vvHalfSpace::contains(const vvVector3& pos)
{
  int i;

  for (i = 0; i < 3; ++i)
  {
    if (_splitPlane->_normal.e[i] < 0.0f)
    {
      if (pos.e[i] < _splitPlane->_point.e[i])
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    if (_splitPlane->_normal.e[i] > 0.0f)
    {
      if (pos.e[i] > _splitPlane->_point.e[i])
      {
        return true;
      }
      else
      {
        return false;
      }
    }
  }
  return false;
}

bool vvHalfSpace::isLeaf() const
{
  return (_firstSon == NULL);
}

void vvHalfSpace::setId(const int id)
{
  _id = id;
}

void vvHalfSpace::setFirstSon(vvHalfSpace* firstSon)
{
  _firstSon = firstSon;
}

void vvHalfSpace::setNextBrother(vvHalfSpace* nextBrother)
{
  _nextBrother = nextBrother;
}

void vvHalfSpace::setSplitPlane(vvPlane* splitPlane)
{
  _splitPlane = splitPlane;
}

void vvHalfSpace::setBricks(std::vector<vvBrick*>* bricks)
{
  vvVector3 minCorner = vvVector3(FLT_MAX, FLT_MAX, FLT_MAX);
  vvVector3 maxCorner = vvVector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  for (std::vector<vvBrick*>::const_iterator it = bricks->begin(); it != bricks->end(); ++it)
  {
    const vvAABB aabb = (*it)->getAABB();
    const vvVector3 minAABB = aabb._bottomLeftBackCorner;
    const vvVector3 maxAABB = aabb._topRightFrontCorner;
    for (int i = 0; i < 3; ++i)
    {
      if (minAABB[i] < minCorner[i])
      {
        minCorner[i] = minAABB[i];
      }
      else if (maxAABB[i] > maxCorner[i])
      {
        maxCorner[i] = maxAABB[i];
      }
    }
  }
  delete _boundingBox;
  _boundingBox = new vvAABB(minCorner, maxCorner);
  _bricks = bricks;
}

void vvHalfSpace::setPercent(const float percent)
{
  _percent = percent;
}

int vvHalfSpace::getId() const
{
  return _id;
}

vvHalfSpace* vvHalfSpace::getFirstSon() const
{
  return _firstSon;
}

vvHalfSpace* vvHalfSpace::getNextBrother() const
{
  return _nextBrother;
}

vvPlane* vvHalfSpace::getSplitPlane() const
{
  return _splitPlane;
}

std::vector<vvBrick*>* vvHalfSpace::getBricks() const
{
  return _bricks;
}

float vvHalfSpace::getPercent() const
{
  return _percent;
}

float vvHalfSpace::getActualPercent() const
{
  return _actualPercent;
}

vvAABB* vvHalfSpace::getBoundingBox() const
{
  return _boundingBox;
}

vvRect* vvHalfSpace::getProjectedScreenRect(const vvVector3* probeMin, const vvVector3* probeMax, const bool recalculate)
{
  if (recalculate)
  {
    delete _projectedScreenRect;
    vvVector3 min;
    vvVector3 max;
    for (int i = 0; i < 3; i++)
    {
      if (_boundingBox->_bottomLeftBackCorner[i] < probeMin->e[i])
      {
        min.e[i] = probeMin->e[i];
      }
      else
      {
        min.e[i] = _boundingBox->_bottomLeftBackCorner[i];
      }

      if (_boundingBox->_topRightFrontCorner[i] > probeMax->e[i])
      {
        max.e[i] = probeMax->e[i];
      }
      else
      {
        max.e[i] = _boundingBox->_topRightFrontCorner[i];
      }
    }
    vvAABB aabb(min, max);
    _projectedScreenRect = aabb.getProjectedScreenRect();
  }
  return _projectedScreenRect;
}

float vvHalfSpace::calcContainedVolume() const
{
  float w, h, d;

  w = h = d = 0.0f;
  for(std::vector<vvBrick*>::iterator it = _bricks->begin(); it != _bricks->end(); ++it)
  {
    vvBrick *tmp = *it;
    w += tmp->getAABB().calcWidth();
    h += tmp->getAABB().calcHeight();
    d += tmp->getAABB().calcDepth();
  }
  return w * h * d;
}

//============================================================================
// vvSpacePartitioner Method Definitions
//============================================================================
vvHalfSpace* vvSpacePartitioner::getAABBHalfSpaces(std::vector<vvBrick*>* bricks,
                                                   const float percent1, const float percent2)
{
  vvHalfSpace* result = new vvHalfSpace[2];

  vvBrick* tmp;
  vvBrick** tmpArray;
  vvVector3 n1, n2;
  vvVector3 pnt;
  float dim[3];
  float min[3];
  float max[3];
  float tmpf;
  int cnt[3];
  int ratio[3][2];
  int bestRatio[3][2];
  float bestWorkLoad[3][2];                       // stored for convenience
  float meanSqrErrorRatio[3];
  int splitAxis;

  // Determine the appropriate axis for the division plane as follows:
  //
  // Let x:y be the desired share (in percent).
  //
  // 1.) Determine w, h and d of the parental volume.
  // 2.) for each axis x, y, z, determine if the volume could be split
  //     so that two half spaces with non overlapping bricks would result
  //     with one half space containing x% and the other one containing y%.
  //     of the bricks.
  // 3.) If 2.) is true for one or more axes, split the volume along this
  //     one / an arbitrary one of these.
  // 4.= If 2.) isn't true, for each axis determine the pair of x':y' values
  //     minimizing the meanSqrError with x:y. Then choose the one minimizing
  //     the overall meanSqrError.

  // Get the aabb for the parent share of the volume.
  max[0] = -FLT_MAX;
  min[0] = FLT_MAX;
  max[1] = -FLT_MAX;
  min[1] = FLT_MAX;
  max[2] = -FLT_MAX;
  min[2] = FLT_MAX;

  tmpArray = new vvBrick*[bricks->size()];
  int i = 0;

  for(std::vector<vvBrick*>::iterator it = bricks->begin();
      it != bricks->end();
      ++it)
  {
    tmp = *it;
    tmpf = tmp->getAABB().calcMaxExtent(vvVector3(1, 0, 0));
    if (tmpf > max[0])
    {
      max[0] = tmpf;
    }

    tmpf = tmp->getAABB().calcMinExtent(vvVector3(1, 0, 0));
    if (tmpf < min[0])
    {
      min[0] = tmpf;
    }

    tmpf = tmp->getAABB().calcMaxExtent(vvVector3(0, 1, 0));
    if (tmpf > max[1])
    {
      max[1] = tmpf;
    }

    tmpf = tmp->getAABB().calcMinExtent(vvVector3(0, 1, 0));
    if (tmpf < min[1])
    {
      min[1] = tmpf;
    }

    tmpf = tmp->getAABB().calcMaxExtent(vvVector3(0, 0, 1));
    if (tmpf > max[2])
    {
      max[2] = tmpf;
    }

    tmpf = tmp->getAABB().calcMinExtent(vvVector3(0, 0, 1));
    if (tmpf < min[2])
    {
      min[2] = tmpf;
    }

    tmpArray[i] = tmp;
    ++i;
  }

  // Get w, h and d.
  for (i = 0; i < 3; ++i)
  {
    dim[i] = max[i] - min[i];
  }

  // Calc the obj count along each axis.
  for (i = 0; i < 3; ++i)
  {
    std::set<float> vals;
    for(std::vector<vvBrick*>::iterator it = bricks->begin();
      it != bricks->end();
      ++it)
    {
      vals.insert((*it)->getAABB().calcCenter().e[i]);
    }
    cnt[i] = vals.size();
  }

  // Reconstruct the 3D grid. This is done since generally the assumption isn't
  // valid that each obj occupies the same volume (determined through its aabb).

  // Sort overall array by x-axis.
  vvBrick::sortByCenter(tmpArray, bricks->size(), vvVector3(1, 0, 0));

  vvBrick*** dimX = new vvBrick**[cnt[0]];

  // Build the first dimension.
  int iterator = 0;
  for (int i = 0; i < cnt[0]; ++i)
  {
    dimX[i] = new vvBrick*[cnt[1] * cnt[2]];
    for (int j = 0; j < cnt[1] * cnt[2]; ++j)
    {
      dimX[i][j] = tmpArray[iterator];
      ++iterator;
    }
  }

  // Sort for second dimension.
  for (int i = 0; i < cnt[0]; ++i)
  {
    vvBrick::sortByCenter(dimX[i], cnt[1]*cnt[2], vvVector3(0, 1, 0));
  }

  // Build second dimension.
  vvBrick**** grid = new vvBrick***[cnt[0]];
  for (int i = 0; i < cnt[0]; ++i)
  {
    grid[i] = new vvBrick**[cnt[1]];
    iterator = 0;
    for (int j = 0; j < cnt[1]; ++j)
    {
      grid[i][j] = new vvBrick*[cnt[2]];
      for (int k = 0; k < cnt[2]; ++k)
      {
        grid[i][j][k] = dimX[i][iterator];
        ++iterator;
      }

      // Sort on the fly.
      vvBrick::sortByCenter(grid[i][j], cnt[2], vvVector3(0, 0, 1));
    }
  }

  // No need for this anymore.
  for (int i = 0; i < cnt[0]; ++i)
  {
    delete[] dimX[i];
  }
  delete[] dimX;


  // Derive the ratios for the three axes respectivly.

  // We will compare the actual workload against this one.
  float idealWorkLoad[2];
  idealWorkLoad[0] = percent1;
  idealWorkLoad[1] = percent2;

  for (i = 0; i < 3; ++i)
  {
    meanSqrErrorRatio[i] = FLT_MAX;

    // Start solution.
    ratio[i][0] = cnt[i];
    ratio[i][1] = 0;
    bestRatio[i][0] = cnt[i];
    bestRatio[i][1] = 0;

    // Iterate over all possible ratios by swapping the
    // greatest share from the left to the right side.

    while (ratio[i][0] >= 0)
    {
      int iteratorX = 0;
      int iteratorY = 0;
      int iteratorZ = 0;

      float workLoad[] = { 0.0f, 0.0f };

      // For left and right work load.
      for (int j = 0; j < 2; ++j)
      {
        for (int k = 0; k < ratio[i][j]; ++k)
        {
          switch (i)
          {
          case 0:
            workLoad[j] += grid[iteratorX][iteratorY][iteratorZ]->getAABB().calcWidth();
            ++iteratorX;
            break;
          case 1:
            workLoad[j] += grid[iteratorX][iteratorY][iteratorZ]->getAABB().calcHeight();
            ++iteratorY;
            break;
          case 2:
            workLoad[j] += grid[iteratorX][iteratorY][iteratorZ]->getAABB().calcDepth();
            ++iteratorZ;
            break;
          default:
            break;
          }
        }

        // Normalize (to 100) the respective work load.
        switch (i)
        {
        case 0:
          workLoad[j] /= dim[0];
          break;
        case 1:
          workLoad[j] /= dim[1];
          break;
        case 2:
          workLoad[j] /= dim[2];
          break;
        default:
          break;
        }
        workLoad[j] *= 100.0f;
      }

      // If the mean sqr error is least, this is the best work load so far.
      float err = vvToolshed::meanAbsError(idealWorkLoad, workLoad, 2);
      if (err < meanSqrErrorRatio[i])
      {
        bestRatio[i][0] = ratio[i][0];
        bestRatio[i][1] = ratio[i][1];
        bestWorkLoad[i][0] = workLoad[0];
        bestWorkLoad[i][1] = workLoad[1];
        meanSqrErrorRatio[i] = err;
      }

      // Iterate on.
      --ratio[i][0];
      ++ratio[i][1];
    }
  }

  // Now find the axis with the smallest mean abs error. This yields
  // the axis along which to split as well as the desired ratio.
  float leastError = FLT_MAX;
  splitAxis = -1;

  for (i = 0; i < 3; ++i)
  {
    if (meanSqrErrorRatio[i] < leastError)
    {
      leastError = meanSqrErrorRatio[i];
      splitAxis = i;
    }
  }

  // Split the volume along the axis.

  // Calculate the splitting planes.
  for (i = 0; i < 3; ++i)
  {
    if (i == splitAxis)
    {
      n1.e[i] = -1;
      n2.e[i] = 1;
      pnt.e[i] = min[i] + bestWorkLoad[i][0] * dim[i] * 0.01f;
      result[0]._actualPercent = bestWorkLoad[i][0];
      result[1]._actualPercent = bestWorkLoad[i][1];
    }
    else
    {
      n1.e[i] = 0;
      n2.e[i] = 0;
      pnt.e[i] = 0;
    }
  }

  result[0].setSplitPlane(new vvPlane(pnt, n1));
  result[1].setSplitPlane(new vvPlane(pnt, n2));

  // Finally distribute pointers to the bricks.
  result[0].setBricks(new std::vector<vvBrick*>());
  result[1].setBricks(new std::vector<vvBrick*>());
  for(std::vector<vvBrick*>::iterator it = bricks->begin();
    it != bricks->end();
    ++it)
  {
    vvBrick *tmp = *it;
    if (tmp->getAABB().calcCenter().e[splitAxis] < pnt.e[splitAxis])
    {
      result[0].getBricks()->push_back(tmp);
    }
    else
    {
      result[1].getBricks()->push_back(tmp);
    }
  }

  // Clean up.
  for (int i = 0; i < cnt[0]; ++i)
  {
    for (int j = 0; j < cnt[1]; ++j)
    {
      delete[] grid[i][j];
    }
    delete[] grid[i];
  }
  delete[] grid;
  delete[] tmpArray;

  return result;
}

//============================================================================
// vvBspTree Method Definitions
//============================================================================

vvBspTree::vvBspTree(float* partitioning, const int length, std::vector<vvBrick*>* bricks)
{
  _root = new vvHalfSpace;
  _root->setPercent(100.0f);
  buildHierarchy(_root, partitioning, length, 0, length - 1);
  _leafs = new std::vector<vvHalfSpace*>();
  distributeBricks(_root, bricks);
  _root->_actualPercent = 100.0f;
  _visitor = NULL;
}

vvBspTree::~vvBspTree()
{
  delete _root;
  delete _leafs;
  delete _visitor;
}

void vvBspTree::traverse(const vvVector3& pos)
{
  traverse(pos, _root);
}

std::vector<vvHalfSpace*>* vvBspTree::getLeafs() const
{
  return _leafs;
}

void vvBspTree::print()
{
  print(_root, 0);
}

void vvBspTree::setVisitor(vvVisitor* visitor)
{
  delete _visitor;
  _visitor = visitor;
}

void vvBspTree::buildHierarchy(vvHalfSpace* node, float* partitioning, const int length,
                               const int startIdx, const int endIdx)
{
  float percent;                                          // Share for this node.
  float percent1, percent2;                               // Share for the two child nodes.
  int startIdx1, startIdx2, endIdx1, endIdx2;
  int length1, length2;
  int i;

  if (length > 1)
  {
    vvHalfSpace* childLeft = new vvHalfSpace();
    vvHalfSpace* childRight = new vvHalfSpace();
    node->addChild(childLeft);
    node->addChild(childRight);

    // Get the indices the 2 children will use to
    // address the percent array values.
    if ((length & 1) == 0)
    {
      length1 = length2 = (length >> 1);
      startIdx1 = startIdx;
      endIdx1 = startIdx+(length >> 1) - 1;
      startIdx2 = startIdx+(length >> 1);
      endIdx2 = endIdx;
    }
    else
    {
      length1 = (length >> 1) + 1;
      length2 = (length >> 1);
      startIdx1 = startIdx;
      endIdx1 = startIdx + (length >> 1);
      startIdx2 = startIdx + (length >> 1) + 1;
      endIdx2 = endIdx;
    }

    // Distribute share to children.
    percent1 = percent2 = 0.0f;
    for (i = startIdx1; i <= endIdx1; ++i)
    {
      percent1 += partitioning[i];
    }

    for (i = startIdx2; i <= endIdx2; ++i)
    {
      percent2 += partitioning[i];
    }

    percent = percent1 + percent2;
    childLeft->setPercent(percent1 / percent * 100);
    childRight->setPercent(percent2 / percent * 100);

    // Do the same thing for both children.
    buildHierarchy(childLeft, partitioning, length1, startIdx1, endIdx1);
    buildHierarchy(childRight, partitioning, length2, startIdx2, endIdx2);
  }
}

void vvBspTree::distributeBricks(vvHalfSpace* node, std::vector<vvBrick*>* bricks)
{
  // No leaf?
  if (node->getFirstSon() != NULL)
  {
    // Only one child?
    if (node->getFirstSon()->getNextBrother() != NULL)
    {
      vvHalfSpace* hs = vvSpacePartitioner::getAABBHalfSpaces(bricks,
                                                              node->getFirstSon()->getPercent(),
                                                              node->getFirstSon()->getNextBrother()->getPercent());
      node->getFirstSon()->_actualPercent = hs[0].getActualPercent();
      node->getFirstSon()->setSplitPlane(new vvPlane(hs[0].getSplitPlane()->_point, hs[0].getSplitPlane()->_normal));
      node->getFirstSon()->getNextBrother()->_actualPercent = hs[1].getActualPercent();
      node->getFirstSon()->getNextBrother()->setSplitPlane(new vvPlane(hs[1].getSplitPlane()->_point, hs[1].getSplitPlane()->_normal));
      distributeBricks(node->getFirstSon(), hs[0].getBricks());
      distributeBricks(node->getFirstSon()->getNextBrother(), hs[1].getBricks());

      // Just delete the array, not the bricks!
      delete[] hs;
    }
  }
  else
  {
    // Leafs store bricks.
    node->setBricks(bricks);
    _leafs->push_back(node);
  }
}

void vvBspTree::print(vvHalfSpace* node, const int indent)
{
  const int inc = 4;
  int i;

  for (i = 0; i < indent; ++i)
  {
    std::cerr << " ";
  }
  std::cerr << "Desired: " << node->getPercent() << "%" << std::endl;
  for (i = 0; i < indent; ++i)
  {
    std::cerr << " ";
  }
  std::cerr << "Realized: " << node->getActualPercent() << "%" << std::endl;
  if (node->isLeaf())
  {
    for (i = 0; i < indent; ++i)
    {
      std::cerr << " ";
    }
    std::cerr << "# bricks: " << node->getBricks()->size() << std::endl;
    for (i = 0; i < indent; ++i)
    {
      std::cerr << " ";
    }
    std::cerr << "contained volume: " << node->calcContainedVolume() << std::endl;
  }

  if (node->getFirstSon() != NULL)
  {
    print(node->getFirstSon(), indent + inc);
  }

  if (node->getNextBrother() != NULL)
  {
    print(node->getNextBrother(), indent);
  }
}

void vvBspTree::traverse(const vvVector3& pos, vvHalfSpace* node)
{
  if (node->isLeaf())
  {
    // Since this bsp tree implementation utilized the visitor
    // pattern, rendering is initiated using the visit() / accept()
    // mechanism of the half space node / its visitor.
    node->accept(_visitor);
  }
  else
  {
    if (node->getFirstSon()->contains(pos))
    {
      traverse(pos, node->getFirstSon()->getNextBrother());
      traverse(pos, node->getFirstSon());
    }
    else
    {
      traverse(pos, node->getFirstSon());
      traverse(pos, node->getFirstSon()->getNextBrother());
    }
  }
}
