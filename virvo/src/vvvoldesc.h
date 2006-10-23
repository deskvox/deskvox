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

#ifndef _VVVOLDESC_H_
#define _VVVOLDESC_H_

#include <stdlib.h>

#include "vvexport.h"
#include "vvtoolshed.h"
#include "vvvecmath.h"
#include "vvtransfunc.h"
#include "vvsllist.h"

//============================================================================
// Class Definition
//============================================================================

/** Volume description.
  The volume description contains basically all the elements which describe
  the volume data. Most of this information can be saved to a file using
  the appropriate file format.
  @author Juergen Schulze-Doebold (schulze@hlrs.de)
*/

#ifndef __APPLE__
template class VIRVOEXPORT vvSLList<uchar*>;
#endif

class VIRVOEXPORT vvVolDesc
{
  public:
    struct line
    {
      int* start;
      int* end;
      float step;
      float depthStep;
      float pos;
    };

    enum ErrorType                                ///  error Codes
    {
      OK,                                         ///< no error
      TYPE_ERROR                                  ///< data type error
    };
    enum AxisType                                 ///  names for coordinate axes
    { X_AXIS, Y_AXIS, Z_AXIS };
    enum InterpolationType                        ///  interpolation types to use for resampling
    {
      NEAREST,                                    ///< nearest neighbor
      TRILINEAR                                   ///< trilinear
    };
    enum DeleteType                               /// types for data deletion
    {
      NO_DELETE,                                  ///< don't delete data because it will be deleted by the caller
      NORMAL_DELETE,                              ///< delete data when not used anymore, use normal delete (delete)
      ARRAY_DELETE                                ///< delete data when not used anymore, use array delete (delete[])
    };
    enum NormalizationType                        /// type of normalization
    {
      VV_LINEAR,                                  ///< linear
      VV_LOGARITHMIC                              ///< logarithmic
    };
    enum MergeType                                /// type of merging two files
    {
      VV_MERGE_SLABS2VOL,                         ///< merge slabs or slices to volume
      VV_MERGE_VOL2ANIM,                          ///< merge volumes to animation
      VV_MERGE_CHAN2VOL                           ///< merge channels
    };
    enum                                          /// size of serialization buffer for attributes [bytes]
    {                                             // 54
      SERIAL_ATTRIB_SIZE = 3*4 + 4 + 1 + 3*4 + 4 + 2*4 + 3*4 + 1
    };
    enum                                          /// icon: bits per pixel (4=RGBA)
    { ICON_BPP = 4};

    enum DiagType
    {
      HISTOGRAM,
      INTENSDIAG
    };

    enum GradientType
    {
      GRADIENT_MAGNITUDE,
      GRADIENT_VECTOR
    };

    enum Channel
    {
      CHANNEL_R = 1,
      CHANNEL_G = 2,
      CHANNEL_B = 4,
      CHANNEL_A = 8
    };
    
    enum BinningType
    {
      LINEAR,
      ISO_DATA,
      OPACITY
    };
    
    static const int DEFAULT_ICON_SIZE;           ///< system default for icon size if not otherwise specified (only stored in XVF files)
    static const int NUM_HDR_BINS;                ///< constant value for HDR transfer functions
    int vox[3];                                   ///< width, height and number of slices of volume [voxels] (0 if no volume loaded)
    int frames;                                   ///< number of animation frames in movie (0 if no volume data stored)
    int bpc;                                      ///< bytes per channel (default = 1); each channel is scalar:<UL>
                                                  ///< <LI>1 = 1 unsigned char</LI>
                                                  ///< <LI>2 = 16 bit unsigned short (12 bit values must be located in 12 most significant bits, padded with 0's)</LI>
                                                  ///< <LI>4 = float</LI>
    int chan;                                     ///< number of channels (default = 1), each channel contains bpc bytes
    float dist[3];                                ///< Distance between sampling points in x/y/z direction [mm]
    float dt;                                     ///< Length of an animation time step [seconds]. Negative values play the animation backwards.
    float real[2];                                ///< 1/2 bpc: physical equivalent of min/max scalar value
                                                  ///< 4 bpc:   min and max values for mapping to transfer function space
    float _scale;                                 ///< determines volume size in conjunction with dist [world space]
    vvVector3 pos;                                ///< location of volume center [mm]
    vvTransFunc tf;                               ///< transfer functions
    int iconSize;                                 ///< width and height of icon [pixels] (0 if no icon stored, e.g., 64 for 64x64 pixels icon)
    uchar* iconData;                              ///< icon image data as RGBA (RGBA, RGBA, ...), starting top left,
                                                  ///< then going right, then down
    int _radius;                                  ///< Radius of the previous sphere
    int* _mask;                                   ///< Mask of the previous sphere
    float* _hdrBinLimits;                         ///< array of bin limits for HDR transfer functions
    BinningType _binning;                         ///< Floating point TF: linear, iso-data, or opacity weighted binning
    bool _transOp;                                ///< true=transfer opacities to bin space

    // Constructors and destructors:
    vvVolDesc();
    vvVolDesc(const char*);
    vvVolDesc(const char*, int, int, int, int, int, int, uchar**, DeleteType=NO_DELETE);
    vvVolDesc(const char*, int, int, int, int, float**);
    vvVolDesc(const char*, int, int, int, int, float**, float**, float**);
    vvVolDesc(const char*, int, int, uchar*);
    vvVolDesc(vvVolDesc*, int=-1);
    virtual ~vvVolDesc();

    // Getters and setters:
    int    getSliceBytes();
    int    getFrameBytes();
    int    getMovieBytes();
    int    getSliceVoxels();
    int    getFrameVoxels();
    int    getMovieVoxels();
    uchar* getRaw(int);
    uchar* getRaw();
    const char* getFilename();
    void   setFilename(const char*);
    void   setCurrentFrame(int);
    int    getCurrentFrame();
    int    getBPV();
    void   setDist(float, float, float);
    void   setDist(vvVector3&);
    vvVector3 getSize();
    int    getStoredFrames();
    float  getValueRange();

    // Conversion routines:
    void   convertBPC(int, bool=false);
    void   convertChannels(int, int frame=-1, bool=false);
    void   deleteChannel(int, bool=false);
    void   bitShiftData(int, int frame=-1, bool=false);
    void   invert();
    void   convertRGB24toRGB8();
    void   flip(AxisType);
    void   rotate(AxisType, int);
    void   convertRGBPlanarToRGBInterleaved(int frame=-1);
    void   toggleEndianness(int frame=-1);
    void   crop(int, int, int, int, int, int);
    void   cropTimesteps(int, int);
    void   resize(int, int, int, InterpolationType, bool=false);
    void   shift(int, int, int);
    void   convertVoxelOrder();
    void   convertCoviseToVirvo();
    void   convertVirvoToCovise();
    void   convertVirvoToOpenGL();
    void   convertOpenGLToVirvo();
    void   makeSphere(int, int, InterpolationType, bool=false);
    void   expandDataRange(bool = false);
    void   zoomDataRange(int, int, int, bool = false);
    void   toggleSign(int frame=-1);
    void   blend(vvVolDesc*, int, bool=false);
    void   swapChannels(int, int, bool=false);
    void   extractChannel(float[3], bool=false);
    bool   makeHeightField(int, int, bool=false);

    // Other routines:
    ErrorType merge(vvVolDesc*, vvVolDesc::MergeType);
    ErrorType mergeFrames();
    void   addFrame(uchar*, DeleteType);
    void   copyFrame(uchar*);
    void   removeSequence();
    void   makeHistogram(int, int, int, int*, int*, float, float);
    void   normalizeHistogram(int, int*, float*, NormalizationType);
    void   makeHistogramTexture(int, int, int, int*, uchar*, NormalizationType, vvColor*, float, float);
    void   createHistogramFiles(bool = false);
    bool   isChannelUsed(int);
    void   makeIcon(int, const uchar*);
    void   makeIcon(int);
    void   printInfoLine(const char* = NULL);
    void   makeInfoString(char*);
    void   makeShortInfoString(char*);
    void   printVolumeInfo();
    void   printStatistics();
    void   printVoxelData(int, int, int=0, int=0);
    void   printHistogram(int, int);
    void   trilinearInterpolation(int, float, float, float, uchar*);
    void   drawBox(int, int, int, int, int, int, int, uchar*);
    void   drawSphere(int, int, int, int, int, uchar*);
    void   drawLine(int, int, int, int, int, int, uchar*);
    void   drawBoundaries(uchar*, int=-1);
    int    serializeAttributes(uchar* = NULL);
    void   deserializeAttributes(uchar*, int=SERIAL_ATTRIB_SIZE);
    void   setSliceData(uchar*, int=0, int=0);
    void   extractSliceData(int, AxisType, int, uchar*);
    void   makeSliceImage(int, AxisType, int, uchar*);
    void   getVolumeSize(AxisType, int&, int&, int&);
    void   deinterlace();
    void   findMinMax(int, float&, float&);
    int    findNumValue(int, float);
    int    findNumUsed(int);
    int    findNumTransparent(int);
    void   calculateDistribution(int, int, float&, float&, float&);
    void   voxelStatistics(int, int, int, int, int, float&, float&);
    float  calculateMean(int);
    float  findClampValue(int, int, float);
    void   computeVolume(int, int, int, int);
    void   resizeEdgeMax(float);
    float  getChannelValue(int, int, int, int, int);
    void   getLineHistData(int, int, int, int, int, int, vvArray<float*>&);
    void   setDefaultRealMinMax();
    void   addGradient(int, GradientType);
    void   addVariance(int);
    void   deleteChannelNames();
    void   setChannelName(int, const char*);
    const char* getChannelName(int);
    void updateFrame(int, uchar*, DeleteType);
    void updateHDRBins(int, bool, bool, bool, BinningType, bool);
    int  findHDRBin(float);
    int  mapFloat2Int(float);
    void makeBinTexture(uchar*, int);
    void computeTFTexture(int, int, int, float*);
    void makeLineTexture(DiagType, uchar, int, int, bool, vvArray<float*>, uchar*);
    void makeLineHistogram(int, int, vvArray<float*>, int*);

  private:
    char*  filename;                              ///< name of volume data file, including extension, excluding path ("" if undefined)
    int    currentFrame;                          ///< current animation frame
    vvSLList<uchar*> raw;                         ///< pointer list to raw volume data
    vvArray<char*> channelNames;                  ///< names of data channels

    void initialize();
    void setDefaults();
    void makeLineIntensDiag(int, vvArray<float*>, int, int*);
    bool isChannelOn(int, unsigned char);
};
#endif

//============================================================================
// End of File
//============================================================================
