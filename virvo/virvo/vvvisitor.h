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

#ifndef VVVISITOR_H
#define VVVISITOR_H

class vvVisitor;

/*!
 * \brief           Visitable class.
 *
 *                  Class that is visitable by a \ref vvVisitor. Inherit this
 *                  class and implement the accept() method in the following
 *                  manner:<br>
 *                  <br>
 *                  <pre>
 *                  void SpecialVisitable::accept(vvVisitor* visitor)
 *                  {
 *                    visitor->visit(this);
 *                  }
 *                  </pre>
 *                  <br>
 *                  Then the visitor can apply logic based upon the type
 *                  of the visitable.
 */
class vvVisitable
{
public:
  virtual ~vvVisitable() {}
  virtual void accept(vvVisitor* visitor) = 0;
private:
};

/*!
 * \brief           Visitor class.
 *
 *                  Inherit this class and implement logic by overriding
 *                  the \ref visit() method. Usually the first thing to
 *                  do their is to cast the \ref vvVisitable object passed
 *                  to \ref visit() to the specialized type and perform
 *                  some operations specific to this type. Thus the logic
 *                  is separated from the data structure containing
 *                  visitable objects as well as from the algorithm using
 *                  them.
 */
class VIRVOEXPORT vvVisitor
{
public:
  virtual ~vvVisitor() {}
  virtual void visit(vvVisitable* obj) const = 0;
private:
};

#endif // VVVISITOR_H
