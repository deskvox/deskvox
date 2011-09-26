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

#ifndef VVBSPTREE_H
#define VVBSPTREE_H

#include "vvvecmath.h"
#include "vvvisitor.h"

#include <vector>

class vvAABB;
class vvBrick;
class vvVisitor;
class vvRect;

typedef std::vector<vvBrick*> BrickList;

/*!
 * \brief           Space node in a bsp tree hierarchy.
 *
 *                  The part of the space this node occupies is propagated to the
 *                  child nodes. If, say, the volume is to be divided
 *                  into 3 sub spaces with weights 33.33%, 33.33% and 33.33%
 *                  respectivelly, a partitioning with the following
 *                  weights will be derived:<br>
 *                  <br>
 *                                  root: 100%<br>
 *                                /             \<br>
 *                            A: 66.66 %     B: 33.33%<br>
 *                              /     \<br>
 *                        C: 50.00%  D: 50.00%<br>
 *                  <br>
 *                  Note that child C and D's share is 50% respectivelly
 *                  rather than 33.33%<br>
 *                  <br>
 *                  If the desired distribution can't be accomodated, an approximation
 *                  minimizing the mean squared error with the desired distribution
 *                  is derived. The resulting share of the volume actually managed
 *                  by this node is stored in the field \ref _actualPercent.
 */
class vvHalfSpace : public vvVisitable
{
  friend class vvSpacePartitioner;
  friend class vvBspTree;
public:
  vvHalfSpace();
  virtual ~vvHalfSpace();

  virtual void accept(vvVisitor* visitor);

  /*!
   * \brief         Add a child node to this half space.
   *
   *                Appends the child.
   * \param         child The child node.
   */
  void addChild(vvHalfSpace* child);
  /*!
   * \brief         Check if a given point is in this half space.
   *
   *                Check if pos is in this half space. Needed
   *                for bsp-tree traversal.
   * \param         pos The point to check this condition for.
   * \return        True if the point is in this half space.
   */
  bool contains(const vvVector3& pos) const;
  /*!
   * \brief         Check if node has no children.
   *
   *                Simple check if this node is a leaf node.
   * \return        true if node is a leaf, false otherwise.
   */
  bool isLeaf() const;
  /*!
   * \brief         Set a distinct integer id.
   *
   *                Ids are useful, e.g. if one wants to render using multiple
   *                threads and later identify which half space is associated
   *                with which thread.
   * \param         id The integer id.
   */
  void setId(int id);
  /*!
   * \brief         Set first son.
   *
   *                Set the first son node.
   * \param         firstSon A pointer to the new first son node.
   */
  void setFirstSon(vvHalfSpace* firstSon);
  /*!
   * \brief         Set next brother.
   *
   *                Set the next brother node.
   * \param         nextBrother A pointer to the new next brother node.
   */
  void setNextBrother(vvHalfSpace* nextBrother);
  /*!
   * \brief         Set the splitting plane.
   *
   *                Set the plane that divides this half space from
   *                the other one. Normal points inwards.
   * \param         splitPlane Splitting plane with inwards pointing normal.
   */
  void setSplitPlane(vvPlane* splitPlane);
  /*!
   * \brief         Set object list.
   *
   *                Set the list of bricks this partial space contains.
   * \param         bricks An array with pointers to convex bricks.
   */
  void setBrickList(const std::vector<BrickList>& brickList);
  /*!
   * \brief         Set percent of parent space this one occupies.
   *
   *                Share of volume data relative to the share of the
   *                parent node.
   * \param         percent The volume share relative to the share of the parent.
   */
  void setPercent(float percent);
  /*!
   * \brief         Get a distinct integer id.
   *
   *                Ids are useful, e.g. if one wants to render using multiple
   *                threads and later identify which half space is associated
   *                with which thread.
   * \return        The integer id.
   */
  int getId() const;
  /*!
   * \brief         Get first son.
   *
   *                Get the first son node.
   * \return        A pointer to the first son node.
   */
  vvHalfSpace* getFirstSon() const;
  /*!
   * \brief         Get next brother.
   *
   *                Get the next brother node.
   * \return        A pointer to the next brother node.
   */
  vvHalfSpace* getNextBrother() const;
  /*!
   * \brief         Get the splitting plane.
   *
   *                Get the plane that divides this half space from
   *                the other one. Normal points inwards.
   * \return        Splitting plane with inwards pointing normal.
   */
  vvPlane* getSplitPlane() const;
  /*!
   * \brief         Get brick list.
   *
   *                Get the list of bricks this partial space contains.
   * \return        An array with pointers to convex bricks.
   */
  std::vector<BrickList>& getBrickList();
  /*!
   * \brief         Get percent of parent space this one occupies.
   *
   *                Share of volume data relative to the share of the
   *                parent node.
   * \return        The volume share relative to the share of the parent.
   */
  float getPercent() const;
  /*!
   * \brief         Get the percent of parent space actually accomodated.
   *
   *                This read-only property stores the share of the parent
   *                volume part that was actually distributed to this node.
   *                When distributing the volume and the desired partitioning
   *                can't be exactly realized, a partitioning minimizing
   *                the mean squared error with the desired one will be
   *                implemented.
   * \return        The actually realized share in percent.
   */
  float getActualPercent() const;
  /*!
   * \brief         Get the bounding box of the half space.
   *
   *                Get an axis aligned bounding box surrounding the bricks
   *                contained within this halfspace.
   * \return        The bounding box around the bricks.
   */
  vvAABB* getBoundingBox() const;
  /*!
   * \brief         Calc the projected screen rect.
   *
   *                Compute a rectangle in scree coordinates completely
   *                occluding the sub-volume as tight as possible.
   *                The rect is stored as a member for later retrieval.
   *                If a probe is present, the bounding box is clipped
   *                against it, otherwise the bounding box is clipped
   *                against the volume.
   *                TODO: Don't clip here, but rather when distributing
   *                the convex bricks.
   * \param         probeMin Used to clip the bounding box.
   * \param         probeMax Used to clip the bounding box.
   * \param         recalculate If true, the member rect will be recalculated.
   * \return        A pointer to the screen rect.
   */
  vvRect* getProjectedScreenRect(const vvVector3* probeMin = 0, const vvVector3* probeMax = 0,
                                 bool recalculate = false);
  /*!
   * \brief         Debug function. Calculate the contained volume.
   *
   *                Calculate the contained volume by evaluating the volume
   *                of the aabbs of the contained bricks. Useful for
   *                debugging, otherwise quite time consuming.
   * \return        The contained volume.
   */
  float calcContainedVolume() const;

  void clipProbe(vvVector3& probeMin, vvVector3& probeMax,
                 vvVector3& probePosObj, vvVector3& probeSizeObj) const;
private:
  int _id;
  vvHalfSpace* _firstSon;
  vvHalfSpace* _nextBrother;

  vvPlane* _splitPlane;
  std::vector<BrickList> _brickList;
  float _percent;
  float _actualPercent;
  vvAABB* _boundingBox;
  vvRect* _projectedScreenRect;
};

/*!
 * \brief           Generic class providing static methods to partition space.
 *
 *                  Space partitioning can be performed in several different ways.
 *                  E.g. space can be partitioned into two separate half spaces.
 *                  Or space is partitioned using another criterium.<br>
 *                  <br>
 *                  Space partitioning can be performed on bricks (aabb's), but
 *                  this isn't necessarily the case. Thus the method used for
 *                  space partitioning is dependent on the brick located in the
 *                  spaces. Thus a generic interface to space partitioning is
 *                  necessary.
 */
class vvSpacePartitioner
{
public:
  /*!
   * \brief         Individual partitioner taking two percent values.
   *
   *                This partitioner will produce to half spaces, each of which
   *                will contain appr. percent1 or percent2 of the provided
   *                bricks respectivelly. Make sure to provide bricks that
   *                are granulare enough to be divided according to the percent
   *                values. Otherwise the percent values will only be approximated.
   *                The provided bricks need to have AABBs. AABBs need to be
   *                partitionable.
   * \param         bricks All bricks to be contained in this half space.
   * \param         percent1 The share for half space 1.
   * \param         percent2 The share for half space 2.
   */
  static vvHalfSpace* getAABBHalfSpaces(const std::vector<BrickList>& bricks,
                                        float percent1, float percent2);
private:
};

/*!
 * \brief           Binary space partitioning tree.
 *
 *                  In order to build up a space partitioning tree, you have to
 *                  provide a pointer to an array of primitives to subdivide
 *                  these. You have to provide an array with a given partition
 *                  with floats ranging from 0.0 to 100.0 (percent).
 */
class vvBspTree
{
public:
  vvBspTree(const float* partitioning, int length, const std::vector<BrickList>& bricks);
  virtual ~vvBspTree();

  void traverse(const vvVector3& pos);

  /*!
   * \brief         Get a list with all leafs.
   *
   *                Get a list with pointers to the leafs.
   * \return        A list with pointers to the leafs.
   */
  std::vector<vvHalfSpace*> *getLeafs() const;
  /*!
   * \brief         Visualize tree using text console.
   *
   *                Print the tree with indented nodes to std::cerr.
   */
  void print();

  /*!
   * \brief         Set the tree's visitor brick.
   *
   *                Tree traversal is realized using the visitor
   *                pattern. The rendering logic is supplied by
   *                an externally implemented visitor class which
   *                essentially will render the contained bricks
   *                based upon the knowledge of their type.
   * \param         visitor The visitor.
   */
  void setVisitor(vvVisitor* visitor);
private:
  vvHalfSpace* _root;
  std::vector<vvHalfSpace*> *_leafs;
  vvVisitor* _visitor;

  /*!
   * \brief         Build up hierarchy of space partitioning nodes.
   *
   *                Builds up the tree given the provided partitioning. Won't
   *                provide the partitioning nodes with primitives, this has to
   *                be performed during a later partitioning step.
   * \param         node The partial space node to append the child to.
   * \param         partitioning An array of (0.0 .. 100.0) values with the partitioning.
   * \param         length The length of the partitioning array.
   * \param         startIdx The start index into the partitioning array.
   * \param         endIdx The end index into the partitioning array.
   */
  void buildHierarchy(vvHalfSpace* node, const float* partitioning,
                      int length, int startIdx, int endIdx);
  void distributeBricks(vvHalfSpace* node, const std::vector<BrickList>& brickList);
  void print(vvHalfSpace* node, int indent);
  void traverse(const vvVector3& pos, vvHalfSpace* node) const;
};

#endif // VVBSPTREE_H
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
