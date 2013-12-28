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

#include <QMouseEvent>

#include "vvlightinteractor.h"

#include <virvo/vvopengl.h>
#include <virvo/vvrect.h>
#include <virvo/private/vvgltools.h>
#include <virvo/private/project.h>

#include <iostream>

vvLightInteractor::vvLightInteractor()
  : _lightingEnabled(true)
  , _mouseButton(Qt::NoButton)
{

}

void vvLightInteractor::render() const
{
  // store GL state
  GLboolean lighting;
  GLboolean depthtest;
  glGetBooleanv(GL_LIGHTING, &lighting);
  glGetBooleanv(GL_DEPTH_TEST, &depthtest);

  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  float r = 3.0f;

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glTranslatef(_pos[0], _pos[1], _pos[2]);
  GLUquadricObj* quad = gluNewQuadric();
  if (_lightingEnabled)
  {
    glColor3f(1.0f, 1.0f, 0.0f);
  }
  else
  {
    glColor3f(0.2f, 0.2f, 0.0f);
  }
  gluSphere(quad, r, 10.0f, 10.0f);

  // axes
  if (hasFocus())
  {
    glBegin(GL_LINES);
      glColor3f(1.0f, 0.0f, 0.0f);
      glVertex3f(r, 0.0f, 0.0f);
      glVertex3f(5 * r, 0.0f, 0.0f);

      glColor3f(0.0f, 1.0f, 0.0f);
      glVertex3f(0.0f, r, 0.0f);
      glVertex3f(0.0f, 5 * r, 0.0f);

      glColor3f(0.0f, 0.0f, 1.0f);
      glVertex3f(0.0f, 0.0f, r);
      glVertex3f(0.0f, 0.0f, 5 * r);
    glEnd();

    GLUquadricObj* quadx = gluNewQuadric();
    glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
    glTranslatef(0.0f, 0.0f, 5 * r);
    glColor3f(1.0f, 0.0f, 0.0f);
    gluCylinder(quadx, r / 2.0f, 0.0f, r * 2.0f, 10.0f, 10.0f);
    glTranslatef(0.0f, 0.0f, -(5 * r));
    glRotatef(-90.0f, 0.0f, 1.0f, 0.0f);

    GLUquadricObj* quady = gluNewQuadric();
    glRotatef(270.0f, 1.0f, 0.0f, 0.0f);
    glTranslatef(0.0f, 0.0f, 5 * r);
    glColor3f(0.0f, 1.0f, 0.0f);
    gluCylinder(quady, r / 2.0f, 0.0f, r * 2.0f, 10.0f, 10.0f);
    glTranslatef(0.0f, 0.0f, -(5 * r));
    glRotatef(-270.0f, 1.0f, 0.0f, 0.0f);

    GLUquadricObj* quadz = gluNewQuadric();
    glTranslatef(0.0f, 0.0f, 5 * r);
    glColor3f(0.0f, 0.0f, 1.0f);
    gluCylinder(quadz, r / 2.0f, 0.0f, r * 2.0f, 10.0f, 10.0f);
    glTranslatef(0.0f, 0.0f, -(5 * r));
  }

  glPopMatrix();

  // reset previous GL state
  if (lighting)
  {
    glEnable(GL_LIGHTING);
  }

  if (depthtest)
  {
    glEnable(GL_DEPTH_TEST);
  }
}

void vvLightInteractor::mouseMoveEvent(QMouseEvent* event)
{
  if (_mouseButton == Qt::LeftButton)
  {
    virvo::Matrix   mv = virvo::gltools::getModelViewMatrix();
    virvo::Matrix   pr = virvo::gltools::getProjectionMatrix();
    virvo::Viewport vp = vvGLTools::getViewport();

    virvo::Vec3 obj;
    virvo::project(&obj, _pos, mv, pr, vp);
    vvVector3 win(event->x(), vp[3] - event->y(), obj[2]);
    virvo::unproject(&_pos, win, mv, pr, vp);
    emit lightPos(_pos);
  }
}

void vvLightInteractor::mousePressEvent(QMouseEvent* event)
{
  _mouseButton = event->button();
}

void vvLightInteractor::mouseReleaseEvent(QMouseEvent*)
{
  if (_mouseButton == Qt::LeftButton)
  {
    emit lightPos(_pos);
  }

  _mouseButton = Qt::NoButton;
}

void vvLightInteractor::setLightingEnabled(bool enabled)
{
  _lightingEnabled = enabled;
}

