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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#if VV_HAVE_NIFTI

#include <virvo/vvvoldesc.h>

#include <nifti/nifti1_io.h>

#include "exceptions.h"
#include "nifti.h"

namespace virvo { namespace nifti {

void load(vvVolDesc* vd)
{
    nifti_image* header = nifti_image_read(vd->getFilename(), 0);

    if (!header)
    {
        throw fileio::exception();
    }

    vd->vox[0] = header->nx;
    vd->vox[1] = header->ny;
    vd->vox[2] = header->nz;

    vd->dist[0] = header->dx;
    vd->dist[1] = header->dy;
    vd->dist[2] = header->dz;

    vd->frames = 1;
    vd->chan = 1;
    vd->bpc = header->nbyper;

    nifti_image* data_section = nifti_image_read(vd->getFilename(), 1);

std::cerr << header->nbyper << std::endl;

    if (!data_section)
    {
        throw fileio::exception();
    }

    uint8_t* raw = new uint8_t[vd->getFrameBytes()];
    memcpy(raw, static_cast<uint8_t*>(data_section->data), vd->getFrameBytes());
    vd->addFrame(raw, vvVolDesc::ARRAY_DELETE);
}

}} // namespace virvo::nifti

#endif // VV_HAVE_NIFTI
