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

#if VV_HAVE_CFITSIO

#include <cassert>
#include <cfloat>
#include <climits>
#include <iostream>
#include <ostream>

#include <fitsio.h>

#include <virvo/vvvoldesc.h>

#include "exceptions.h"
#include "fits.h"

#include "limits"

namespace virvo { namespace fits {

void load(vvVolDesc *vd)
{
    bool verbose = true;
    int numHDUs = 0, hduType = 0, status = 0;
    int naxis = 0, bitpix;
    long naxes[9] = {1,1,1,1,1,1,1,1,1};
    long totpix = 0;

    //open fits file and read structure info -----------------------------
    fitsfile *fptr;
    fits_open_file(&fptr, vd->getFilename(), READONLY, &status);
    if(status != 0)
    {
        fits_report_error(stderr, status);
        return;
    }

    //get the number of HEADER-DATA-UNITS (HDUs)
    fits_get_num_hdus(fptr, &numHDUs, &status);
    if(numHDUs == 0)
    {
        printf("load fits:: No HDUs in fits file: %s \n", vd->getFilename());
    }    
    if(verbose) printf("load fits:: found %d HDUs. Loading only the first! \n", numHDUs);

    //Check if primary HDU is of type IMAGE_HDU
    fits_get_hdu_type(fptr, &hduType, &status);
    if(hduType != IMAGE_HDU) {
        if(verbose) printf("load fits: first HDu is not an image hdu \n");
        return;
    }
    if(verbose) printf("load fits:: first HDU is image HDU \n");

    //Get Infos about current HDU
    fits_get_img_param(fptr, 9, &bitpix, &naxis, naxes, &status);
    totpix = naxes[0] * naxes[1] * naxes[2] * naxes[3] * naxes[4]
       * naxes[5] * naxes[6] * naxes[7] * naxes[8];
    if(verbose)
    {
        printf("load fits:: found %d axes \n", naxis);
        for(int i = 0; i < naxis; i++)
        {
            printf("load fits:: axis %d has size %ld \n", i, naxes[i]);
        }
    }

    //actually read data from file and close it
    double* array_double = new double[totpix];
    float* array_float = new float[totpix];
    assert(array_double);
    assert(array_float);

    long fpixel[3] = {1,1,1};
    int retVal = fits_read_pix(fptr, TDOUBLE, fpixel, totpix, 0, array_double, 0, &status);
    if(verbose) printf("load fits:: retVal fom read_pix is %d   status: %d \n", retVal, status);

    fits_close_file(fptr, &status);

    //crunch numbers, turn double to float, take log10, compute min and max
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();

    for(long i = 0; i < totpix; i++){
        array_float[i] = (float) log10(array_double[i]);

        if(array_float[i] < min) min = array_float[i];
        if(array_float[i] > max) max = array_float[i];
    }

    //the doubles are not needed any more
    delete[] array_double;

    //write data to Volume Description (vd)
    vd->vox[0] = naxes[0];
    vd->vox[1] = naxes[1];
    vd->vox[2] = naxes[2];

    vd->setDist(1.0f,1.0f,1.0f);
    vd->frames = 1;
    vd->bpc = 4;
    vd->setChan(1);

    vd->addFrame(reinterpret_cast<uint8_t*>(array_float), vvVolDesc::ARRAY_DELETE);
    vd->findAndSetRange();

    if(verbose) printf("load fits:: done loading. min: %f   max: %f \n", min, max);

    return;
}

}}  //end of namespace

#endif //VV_HAVE_CFITSIO
