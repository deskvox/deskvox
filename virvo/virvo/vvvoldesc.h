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
#include <vector>

#include "vvexport.h"
#include "vvinttypes.h"
#include "vvvecmath.h"
#include "vvtransfunc.h"
#include "vvsllist.h"
#include "vvarray.h"

template <typename T>
class vvBaseAABB;
typedef vvBaseAABB<float> vvAABBf;
typedef vvAABBf vvAABB;

//============================================================================
// Class Definition
//============================================================================

/** Volume description.
  The volume description contains basically all the elements which describe
  the volume data. Most of this information can be saved to a file using
  the appropriate file format.
  @author Juergen Schulze-Doebold (schulze@hlrs.de)
*/

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
    
    static const size_t DEFAULT_ICON_SIZE;        ///< system default for icon size if not otherwise specified (only stored in XVF files)
    static const size_t NUM_HDR_BINS;             ///< constant value for HDR transfer functions
    vvsize3 vox;                                  ///< width, height and number of slices of volume [voxels] (0 if no volume loaded)
    size_t frames;                                ///< number of animation frames in movie (0 if no volume data stored)
    size_t bpc;                                   ///< bytes per channel (default = 1); each channel is scalar:<UL>
                                                  ///< <LI>1 = 1 unsigned char</LI>
                                                  ///< <LI>2 = 16 bit unsigned short (12 bit values must be located in 12 most significant bits, padded with 0's)</LI>
                                                  ///< <LI>4 = float</LI>
    size_t chan;                                  ///< number of channels (default = 1), each channel contains bpc bytes
    vvVector3 dist;                               ///< Distance between sampling points in x/y/z direction [mm]
    float dt;                                     ///< Length of an animation time step [seconds]. Negative values play the animation backwards.
    float real[2];                                ///< 1/2 bpc: physical equivalent of min/max scalar value
                                                  ///< 4 bpc:   min and max values for mapping to transfer function space
    float _scale;                                 ///< determines volume size in conjunction with dist [world space]
    vvVector3 pos;                                ///< location of volume center [mm]
    vvTransFunc tf;                               ///< transfer functions
    size_t iconSize;                              ///< width and height of icon [pixels] (0 if no icon stored, e.g., 64 for 64x64 pixels icon)
    uint8_t* iconData;                            ///< icon image data as RGBA (RGBA, RGBA, ...), starting top left,
                                                  ///< then going right, then down
    int _radius;                                  ///< Radius of the previous sphere
    int* _mask;                                   ///< Mask of the previous sphere
    float* _hdrBinLimits;                         ///< array of bin limits for HDR transfer functions
    BinningType _binning;                         ///< Floating point TF: linear, iso-data, or opacity weighted binning
    bool _transOp;                                ///< true=transfer opacities to bin space

    // Constructors and destructors:
    vvVolDesc();
    vvVolDesc(const char*);
    vvVolDesc(const char* fn, size_t w, size_t h, size_t s, size_t f, size_t b, size_t m, uint8_t** d, DeleteType=NO_DELETE);
    vvVolDesc(const char* fn, size_t w, size_t h, size_t s, size_t f, float**);
    vvVolDesc(const char* fn, size_t w, size_t h, size_t s, size_t f, float**, float**, float**);
    vvVolDesc(const char* fn, size_t w, size_t h, uint8_t* d);
    vvVolDesc(vvVolDesc*, int=-1);
    virtual ~vvVolDesc();

    // Getters and setters:
    size_t getSliceBytes() const;
    size_t getFrameBytes() const;
    size_t getMovieBytes() const;
    size_t getSliceVoxels() const;
    size_t getFrameVoxels() const;
    size_t getMovieVoxels() const;
    void getBoundingBox(vvAABB& aabb) const;
    /** Legacy function, returns current frame
     @param  default to -1 in the past meant current frame */
    uint8_t* getRaw(int frame = -1) const;
    uint8_t* getRaw(size_t) const;
    const char* getFilename() const;
    void   setFilename(const char*);
    void   setCurrentFrame(size_t);
    size_t getCurrentFrame() const;
    size_t getBPV() const;
    void   setDist(float, float, float);
    void   setDist(const vvVector3& d);
    vvVector3 getSize() const;
    size_t getStoredFrames() const;
    float  getValueRange() const;

    // Conversion routines:
    void   convertBPC(size_t, bool=false);
    void   convertChannels(size_t, int frame=-1, bool=false);
    void   deleteChannel(size_t, bool=false);
    void   bitShiftData(int, int frame=-1, bool=false);
    void   invert();
    void   convertRGB24toRGB8();
    void   flip(vvVecmath::AxisType);
    void   rotate(vvVecmath::AxisType, int);
    void   convertRGBPlanarToRGBInterleaved(int frame=-1);
    void   toggleEndianness(int frame=-1);
    void   crop(size_t, size_t, size_t, size_t, size_t, size_t);
    void   cropTimesteps(size_t start, size_t steps);
    void   resize(size_t, size_t, size_t, InterpolationType, bool=false);
    void   shift(int, int, int);
    void   convertVoxelOrder();
    void   convertCoviseToVirvo();
    void   convertVirvoToCovise();
    void   convertVirvoToOpenGL();
    void   convertOpenGLToVirvo();
    void   makeSphere(size_t outer, size_t inner, InterpolationType, bool=false);
    void   expandDataRange(bool = false);
    void   zoomDataRange(int, int, int, bool = false);
    void   toggleSign(int frame=-1);
    void   blend(vvVolDesc*, int, bool=false);
    void   swapChannels(size_t, size_t, bool=false);
    void   extractChannel(float[3], bool=false);
    bool   makeHeightField(size_t, int, bool=false);

    // Other routines:
    ErrorType merge(vvVolDesc*, vvVolDesc::MergeType);
    ErrorType mergeFrames();
    void   addFrame(uint8_t*, DeleteType, int fd=-1);
    void   copyFrame(uint8_t*);
    void   removeSequence();
    void   makeHistogram(int, size_t, size_t, int*, int*, float, float);
    void   normalizeHistogram(int, int*, float*, NormalizationType);
    void   makeHistogramTexture(int frame, size_t, size_t, size_t*, uint8_t* data, NormalizationType, vvColor*, float, float);
    void   createHistogramFiles(bool = false);
    bool   isChannelUsed(size_t);
    void   makeIcon(size_t, const uint8_t*);
    void   makeIcon(size_t);
    void   printInfoLine(const char* = NULL);
    void   makeInfoString(std::string* infoString);
    void   makeShortInfoString(char*);
    void   printVolumeInfo();
    void   printStatistics();
    void   printVoxelData(int, size_t, size_t=0, size_t=0);
    void   printHistogram(int frame, size_t channel);
    void   trilinearInterpolation(size_t f, float, float, float, uint8_t*);
    void   drawBox(size_t, size_t, size_t, size_t, size_t, size_t, size_t, uint8_t*);
    void   drawSphere(size_t, size_t, size_t, size_t, size_t, uint8_t*);
    void   drawLine(size_t, size_t, size_t, size_t, size_t, size_t, uint8_t*);
    void   drawBoundaries(uchar*, int=-1);
    size_t serializeAttributes(uint8_t* = NULL) const;
    void   deserializeAttributes(uint8_t*, size_t bufSize=SERIAL_ATTRIB_SIZE);
    void   setSliceData(uint8_t*, int=0, int=0);
    void   extractSliceData(int, vvVecmath::AxisType, size_t slice, uint8_t*);
    void   makeSliceImage(int, vvVecmath::AxisType, size_t slice, uint8_t*);
    void   getVolumeSize(vvVecmath::AxisType, size_t&, size_t&, size_t&);
    void   deinterlace();
    void   findMinMax(size_t channel, float&, float&);
    int    findNumValue(int, float);
    int    findNumUsed(size_t channel);
    int    findNumTransparent(int);
    void   calculateDistribution(int frame, size_t chan, float&, float&, float&);
    void   voxelStatistics(size_t, size_t, size_t, size_t, size_t, float&, float&);
    float  calculateMean(int);
    float  findClampValue(int, size_t channel, float);
    void   computeVolume(int, size_t, size_t, size_t);
    void   resizeEdgeMax(float);
    float  getChannelValue(int frame, size_t x, size_t y, size_t z, size_t chan);
    void   getLineHistData(int, int, int, int, int, int, vvArray<float*>&);
    void   setDefaultRealMinMax();
    void   addGradient(size_t srcChan, GradientType);
    void   addVariance(size_t srcChan);
    void   deleteChannelNames();
    void   setChannelName(size_t, const char*);
    const char* getChannelName(size_t);
    void updateFrame(int, uint8_t*, DeleteType);
    void updateHDRBins(size_t numValues, bool, bool, bool, BinningType, bool);
    int  findHDRBin(float);
    int  mapFloat2Int(float);
    void makeBinTexture(uint8_t* texture, size_t width);
    void computeTFTexture(int, int, int, float*);
    void makeLineTexture(DiagType, uchar, int, int, bool, vvArray<float*>, uint8_t*);
    void makeLineHistogram(size_t channel, int buckets, vvArray<float*>, int*);
    void computeMinMaxArrays(uint8_t *minArray, uchar *maxArray, size_t downsample, size_t channel=0, int frame=-1) const;
    vvsize3 voxelCoords(const vvVector3& objCoords) const;
    vvVector3 objectCoords(const vvsize3& voxCoords) const;

  private:
    char*  filename;                              ///< name of volume data file, including extension, excluding path ("" if undefined)
    size_t currentFrame;                          ///< current animation frame
    mutable vvSLList<uint8_t*> raw;               ///< pointer list to raw volume data - mutable because of Java style iterators
    std::vector<size_t> rawFrameNumber;           ///< frame numbers (if frames do not come in sequence)
    vvArray<char*> channelNames;                  ///< names of data channels

    void initialize();
    void setDefaults();
    void makeLineIntensDiag(size_t channel, vvArray<float*>, size_t numValues, int*);
    bool isChannelOn(size_t num, unsigned char);
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
