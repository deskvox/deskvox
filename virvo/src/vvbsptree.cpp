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

#include <set>

#include "vvbsptree.h"
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
  return calcMaxExtend(vvVector3(1, 0, 0)) - calcMinExtend(vvVector3(1, 0, 0));
}

float vvAABB::calcHeight() const
{
  return calcMaxExtend(vvVector3(0, 1, 0)) - calcMinExtend(vvVector3(0, 1, 0));
}

float vvAABB::calcDepth() const
{
  return calcMaxExtend(vvVector3(0, 0, 1)) - calcMinExtend(vvVector3(0, 0, 1));
}

const vvBoxCorners &vvAABB::calcVertices()
{
  for(int i=0; i<8; ++i)
  {
    // return the edges in the necessary order
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

float vvAABB::calcMinExtend(const vvVector3& axis) const
{
  return _bottomLeftBackCorner.e[0] * axis.e[0]
       + _bottomLeftBackCorner.e[1] * axis.e[1]
       + _bottomLeftBackCorner.e[2] * axis.e[2];
}

float vvAABB::calcMaxExtend(const vvVector3& axis) const
{
  return _topRightFrontCorner.e[0] * axis.e[0]
       + _topRightFrontCorner.e[1] * axis.e[1]
       + _topRightFrontCorner.e[2] * axis.e[2];
}

vvVector3 vvAABB::calcCenter() const
{
  return vvVector3((_bottomLeftBackCorner.e[0] + _topRightFrontCorner.e[0]) / 2,
                   (_bottomLeftBackCorner.e[1] + _topRightFrontCorner.e[1]) / 2,
                   (_bottomLeftBackCorner.e[2] + _topRightFrontCorner.e[2]) / 2);
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

//============================================================================
// vvConvexObj Method Definitions
//============================================================================

void vvConvexObj::sortByCenter(vvConvexObj** objects, const int numObjects,
                               const vvVector3& axis)
{
  vvConvexObj* tmp;
  vvConvexObj* tmp2;
  vvConvexObj* tmp3;
  vvVector3* axisGetter;
  int i, j, k;
  int a;

  axisGetter = new vvVector3(0, 1, 2);
  a = axis.dot(axisGetter);
  delete axisGetter;

  // Selection sort.
  for (i = 0; i < numObjects; ++i)
  {
    for (j = i; j < numObjects; ++j)
    {
      tmp = objects[j];
      for (k = i + 1; k < numObjects; ++k)
      {
        tmp2 = objects[k];
        if (tmp->getAABB().calcCenter().e[a] > tmp2->getAABB().calcCenter().e[a])
        {
          tmp3 = objects[j];
          objects[j] = objects[k];
          objects[k] = tmp3;
        }
      }
    }
  }
}

//============================================================================
// vvHalfSpace Method Definitions
//============================================================================

vvHalfSpace::vvHalfSpace()
{
  _firstSon = NULL;
  _nextBrother = NULL;

  _splitPlane = NULL;
  _objects = NULL;
}

vvHalfSpace::~vvHalfSpace()
{
  delete _splitPlane;
  delete _firstSon;
  delete _nextBrother;
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

void vvHalfSpace::setObjects(std::vector<vvConvexObj*>* objects)
{
  _objects = objects;
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

std::vector<vvConvexObj*>* vvHalfSpace::getObjects() const
{
  return _objects;
}

float vvHalfSpace::getPercent() const
{
  return _percent;
}

float vvHalfSpace::getActualPercent() const
{
  return _actualPercent;
}

float vvHalfSpace::calcContainedVolume() const
{
  float w, h, d;

  w = h = d = 0.0f;
  for(std::vector<vvConvexObj *>::iterator it = _objects->begin(); it != _objects->end(); ++it)
  {
    vvConvexObj *tmp = *it;
    w += tmp->getAABB().calcWidth();
    h += tmp->getAABB().calcHeight();
    d += tmp->getAABB().calcDepth();
  }
  return w * h * d;
}

//============================================================================
// vvSpacePartitioner Method Definitions
//============================================================================
vvHalfSpace* vvSpacePartitioner::getAABBHalfSpaces(std::vector<vvConvexObj *> *objects,
                                                   const float percent1, const float percent2)
{
  vvHalfSpace* result = new vvHalfSpace[2];

  vvConvexObj* tmp;
  vvConvexObj** tmpArray;
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
  int i, j, k;

  // Determine the appropriate axis for the division plane as follows:
  //
  // Let x:y be the desired share (in percent).
  //
  // 1.) Determine w, h and d of the parental volume.
  // 2.) for each axis x, y, z, determine if the volume could be split
  //     so that two half spaces with non overlapping bricks would result
  //     with one half space containing x% and the other one containing y%.
  //     of the objects.
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

  tmpArray = new vvConvexObj*[objects->size()];
  i = 0;

  for(std::vector<vvConvexObj *>::iterator it = objects->begin();
      it != objects->end();
      ++it)
  {
    tmp = *it;
    tmpf = tmp->getAABB().calcMaxExtend(vvVector3(1, 0, 0));
    if (tmpf > max[0])
    {
      max[0] = tmpf;
    }

    tmpf = tmp->getAABB().calcMinExtend(vvVector3(1, 0, 0));
    if (tmpf < min[0])
    {
      min[0] = tmpf;
    }

    tmpf = tmp->getAABB().calcMaxExtend(vvVector3(0, 1, 0));
    if (tmpf > max[1])
    {
      max[1] = tmpf;
    }

    tmpf = tmp->getAABB().calcMinExtend(vvVector3(0, 1, 0));
    if (tmpf < min[1])
    {
      min[1] = tmpf;
    }

    tmpf = tmp->getAABB().calcMaxExtend(vvVector3(0, 0, 1));
    if (tmpf > max[2])
    {
      max[2] = tmpf;
    }

    tmpf = tmp->getAABB().calcMinExtend(vvVector3(0, 0, 1));
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
    for(std::vector<vvConvexObj *>::iterator it = objects->begin();
      it != objects->end();
      ++it)
    {
      vals.insert((*it)->getAABB().calcCenter().e[i]);
    }
    cnt[i] = vals.size();
  }

  // Reconstruct the 3D grid. This is done since generally the assumption isn't
  // valid that each obj occupies the same volume (determined through its aabb).

  // Sort overall array by x-axis.
  vvConvexObj::sortByCenter(tmpArray, objects->size(), vvVector3(1, 0, 0));

  vvConvexObj*** dimX = new vvConvexObj**[cnt[0]];

  // Build the first dimension.
  int iterator = 0;
  for (i = 0; i < cnt[0]; ++i)
  {
    dimX[i] = new vvConvexObj*[cnt[1] * cnt[2]];
    for (j = 0; j < cnt[1] * cnt[2]; ++j)
    {
      dimX[i][j] = tmpArray[iterator];
      ++iterator;
    }
  }

  // Sort for second dimension.
  for (i = 0; i < cnt[0]; ++i)
  {
    vvConvexObj::sortByCenter(dimX[i], cnt[1]*cnt[2], vvVector3(0, 1, 0));
  }

  // Build second dimension.
  vvConvexObj**** grid = new vvConvexObj***[cnt[0]];
  for (i = 0; i < cnt[0]; ++i)
  {
    grid[i] = new vvConvexObj**[cnt[1]];
    iterator = 0;
    for (j = 0; j < cnt[1]; ++j)
    {
      grid[i][j] = new vvConvexObj*[cnt[2]];
      for (k = 0; k < cnt[2]; ++k)
      {
        grid[i][j][k] = dimX[i][iterator];
        ++iterator;
      }

      // Sort on the fly.
      vvConvexObj::sortByCenter(grid[i][j], cnt[2], vvVector3(0, 0, 1));
    }
  }

  // No need for this anymore.
  for (i = 0; i < cnt[0]; ++i)
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

      float workLoad[2];
      workLoad[0] = 0.0f;
      workLoad[1] = 0.0f;

      // For left and right work load.
      for (j = 0; j < 2; ++j)
      {
        for (k = 0; k < ratio[i][j]; ++k)
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

  // Finally distribute pointers to the objects.
  result[0].setObjects(new std::vector<vvConvexObj*>());
  result[1].setObjects(new std::vector<vvConvexObj*>());
  for(std::vector<vvConvexObj *>::iterator it = objects->begin();
    it != objects->end();
    ++it)
  {
    vvConvexObj *tmp = *it;
    if (tmp->getAABB().calcCenter().e[splitAxis] < pnt.e[splitAxis])
    {
      result[0].getObjects()->push_back(tmp);
    }
    else
    {
      result[1].getObjects()->push_back(tmp);
    }
  }

  // Clean up.
  for (i = 0; i < cnt[0]; ++i)
  {
    for (j = 0; j < cnt[1]; ++j)
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

vvBspTree::vvBspTree(float* partitioning, const int length, std::vector<vvConvexObj*>* objects)
{
  _root = new vvHalfSpace;
  _root->setPercent(100.0f);
  buildHierarchy(_root, partitioning, length, 0, length - 1);
  _leafs = new std::vector<vvHalfSpace*>();
  distributeObjects(_root, objects);
  _root->_actualPercent = 100.0f;
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
    if ((length % 2) == 0)
    {
      length1 = length2 = length / 2;
      startIdx1 = startIdx;
      endIdx1 = startIdx+length/2-1;
      startIdx2 = startIdx+length/2;
      endIdx2 = endIdx;
    }
    else
    {
      length1 = length/2+1;
      length2 = length/2;
      startIdx1 = startIdx;
      endIdx1 = startIdx+length/2;
      startIdx2 = startIdx+length/2+1;
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

void vvBspTree::distributeObjects(vvHalfSpace* node, std::vector<vvConvexObj*>* objects)
{
  // No leaf?
  if (node->getFirstSon() != NULL)
  {
    // Only one child?
    if (node->getFirstSon()->getNextBrother() != NULL)
    {
      vvHalfSpace* hs = vvSpacePartitioner::getAABBHalfSpaces(objects,
                                                              node->getFirstSon()->getPercent(),
                                                              node->getFirstSon()->getNextBrother()->getPercent());
      node->getFirstSon()->_actualPercent = hs[0].getActualPercent();
      node->getFirstSon()->setSplitPlane(new vvPlane(hs[0].getSplitPlane()->_point, hs[0].getSplitPlane()->_normal));
      node->getFirstSon()->getNextBrother()->_actualPercent = hs[1].getActualPercent();
      node->getFirstSon()->getNextBrother()->setSplitPlane(new vvPlane(hs[1].getSplitPlane()->_point, hs[1].getSplitPlane()->_normal));
      distributeObjects(node->getFirstSon(), hs[0].getObjects());
      distributeObjects(node->getFirstSon()->getNextBrother(), hs[1].getObjects());

      // Just delete the array, not the objects!
      delete[] hs;
    }
  }
  else
  {
    // Leafs store objects.
    node->setObjects(objects);
    _leafs->push_back(node);
  }
}

void vvBspTree::print(vvHalfSpace* node, const int indent)
{
  int inc = 4;
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
    std::cerr << "# objects: " << node->getObjects()->size() << std::endl;
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
