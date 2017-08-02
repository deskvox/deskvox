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

#include <cassert>

#include <nifti1_io.h>

#include <virvo/vvvoldesc.h>

#include "exceptions.h"
#include "nifti.h"

namespace virvo { namespace nifti {

void load(vvVolDesc* vd)
{
    // read nifti header ----------------------------------

    nifti_image* header = nifti_image_read(vd->getFilename(), 0);

    if (!header)
    {
        throw fileio::exception();
    }


    // dimensions

    vd->vox[0] = header->nx;
    vd->vox[1] = header->ny;
    vd->vox[2] = header->nz;

    vd->dist[0] = header->dx;
    vd->dist[1] = header->dy;
    vd->dist[2] = header->dz;

    // no support for animation

    vd->frames = 1;

    bool is_signed_int = false;

    // bytes per pixel and num channels

    switch (header->datatype)
    {
    case NIFTI_TYPE_RGB24:
        vd->chan = 3;
        vd->bpc = header->nbyper / 3;
        break;
    case NIFTI_TYPE_RGBA32:
        vd->chan = 4;
        vd->bpc = header->nbyper / 4;
        break;
    case NIFTI_TYPE_INT8:
    case NIFTI_TYPE_INT16:
    case NIFTI_TYPE_INT32:
        is_signed_int = true;
        // fall through
    case NIFTI_TYPE_UINT8:
    case NIFTI_TYPE_UINT16:
    case NIFTI_TYPE_UINT32:
    case NIFTI_TYPE_FLOAT32:
        // all: fall through
    default:
        vd->chan = 1;
        vd->bpc = header->nbyper;
        break;
    }

    // voxel properties

    vd->setSignedInt(is_signed_int);
    vd->setVoxelScale(header->scl_slope);
    vd->setVoxelOffset(header->scl_inter);


    // data range

    for (size_t c = 0; c < vd->chan; ++c)
    {
        vd->real[c][0] = header->cal_min;
        vd->real[c][1] = header->cal_max;
    }


    // read image data ------------------------------------

    nifti_image* data_section = nifti_image_read(vd->getFilename(), 1);


    if (!data_section)
    {
        throw fileio::exception();
    }

    uint8_t* raw = new uint8_t[vd->getFrameBytes()];
    memcpy(raw, static_cast<uint8_t*>(data_section->data), vd->getFrameBytes());
    vd->addFrame(raw, vvVolDesc::ARRAY_DELETE);


    // find min/max if cal_{min|max} zero

    for (size_t c = 0; c < vd->chan; ++c)
    {
        if (vd->real[c][0] == 0.f && vd->real[c][1] == 0.f)
        {
            if (vd->bpc == 1 || vd->bpc == 2)
                vd->real[c][1] = 1.f;
            else if (vd->bpc == 4) // floating point: search for min/max in data
                vd->findMinMax(c, vd->real[c][0], vd->real[c][1]);
            else
                assert(0);
        }
    }
}

void save(const vvVolDesc* vd)
{
    int dims[] = { 3, (int)vd->vox[0], (int)vd->vox[1], (int)vd->vox[2], 1, 0, 0, 0 };

    int datatype = 0;
    if (vd->bpc == 1)
        datatype = NIFTI_TYPE_UINT8;
    else if (vd->bpc == 2)
        datatype = NIFTI_TYPE_UINT16;
    else if (vd->bpc == 4)
        datatype = NIFTI_TYPE_FLOAT32;

    nifti_image* img = nifti_make_new_nim(dims, datatype, 1);

    nifti_set_filenames(img, vd->getFilename(), 0, 0);

    img->dx = vd->dist[0];
    img->dy = vd->dist[1];
    img->dz = vd->dist[2];

    img->data = vd->getRaw();

    nifti_image_write(img);
}

}} // namespace virvo::nifti

#endif // VV_HAVE_NIFTI
