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

#ifndef VV_VOLDESC_H
#define VV_VOLDESC_H

#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/split_member.hpp>

#include <stdlib.h>
#include <string>
#include <vector>

#include "math/math.h"

#include "vvexport.h"
#include "vvinttypes.h"
#include "vvtransfunc.h"
#include "vvsllist.h"

//============================================================================
// Class Definition
//============================================================================

/** Volume description.
  The volume description contains basically all the elements which describe
  the volume data. Most of this information can be saved to a file using
  the appropriate file format.

  Raw data is stored in host byte order.

  @author Juergen Schulze-Doebold (schulze@hlrs.de)
*/

class VIRVO_FILEIOEXPORT vvVolDesc
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
    virvo::vector< 3, ssize_t > vox;              ///< width, height and number of slices of volume [voxels] (0 if no volume loaded)
    size_t frames;                                ///< number of animation frames in movie (0 if no volume data stored)
    size_t bpc;                                   ///< bytes per channel (default = 1); each channel is scalar:<UL>
                                                  ///< <LI>1 = 1 unsigned char</LI>
                                                  ///< <LI>2 = 16 bit unsigned short (12 bit values must be located in 12 most significant bits, padded with 0's)</LI>
                                                  ///< <LI>4 = float</LI>
    size_t chan;                                  ///< number of channels (default = 1), each channel contains bpc bytes
    virvo::vec3f dist;                            ///< Distance between sampling points in x/y/z direction [mm]
    float dt;                                     ///< Length of an animation time step [seconds]. Negative values play the animation backwards.
    std::vector<virvo::vec2> real;                ///< 1/2 bpc: physical equivalent of min/max scalar value
                                                  ///< 4 bpc:   min and max values for mapping to transfer function space
                                                  ///<     if more than 1: each channel has its own
    std::vector<float> channelWeights;            ///< scalar weight for each channel, default: 1.0f
    float _scale;                                 ///< determines volume size in conjunction with dist [world space]
    virvo::vec3f pos;                             ///< location of volume center [mm]
    std::vector<vvTransFunc> tf;                  ///< transfer functions (if more than 1: each channel has its own)
    size_t iconSize;                              ///< width and height of icon [pixels] (0 if no icon stored, e.g., 64 for 64x64 pixels icon)
    uint8_t* iconData;                            ///< icon image data as RGBA (RGBA, RGBA, ...), starting top left,
                                                  ///< then going right, then down
    int _radius;                                  ///< Radius of the previous sphere
    int* _mask;                                   ///< Mask of the previous sphere
    float* _hdrBinLimits;                         ///< array of bin limits for HDR transfer functions
    BinningType _binning;                         ///< Floating point TF: linear, iso-data, or opacity weighted binning
    bool _transOp;                                ///< true=transfer opacities to bin space

    //--- serialization ----------------------------------------------------------------------------

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    template<class A>
    void save(A& a, unsigned /*version*/) const
    {
      // Header (from vvVolDesc::serializeAttributes)
      a & vox;
      a & frames;
      a & bpc;
      a & dist;
      a & dt;
      a & real;
      a & pos;
      a & chan;
      a & _scale;

      // Data (from vvSocketIO::putVolume)
      size_t size = getFrameBytes();

      for (size_t k = 0; k < frames; k++)
      {
        a & boost::serialization::make_binary_object(getRaw(k), size);
      }
    }

    template<class A>
    void load(A& a, unsigned /*version*/)
    {
      // Header (from vvVolDesc::deserializeAttributes)
      a & vox;
      a & frames;
      a & bpc;
      a & dist;
      a & dt;
      a & real;
      a & pos;
      a & chan;
      a & _scale;

      // Data (from vvSocketIO::getVolume)
      size_t size = getFrameBytes();

      for (size_t k = 0; k < frames; ++k)
      {
        uint8_t* buffer = new uint8_t[size];

        a & boost::serialization::make_binary_object(buffer, size);

        addFrame(buffer, vvVolDesc::ARRAY_DELETE);
      }
    }

    //----------------------------------------------------------------------------------------------

    // Constructors and destructors:
    vvVolDesc();
    vvVolDesc(const char*);
    vvVolDesc(const char* fn, size_t w, size_t h, size_t s, size_t f, size_t b, size_t m, uint8_t** d, DeleteType=NO_DELETE);
    vvVolDesc(const char* fn, size_t w, size_t h, size_t s, size_t f, float**);
    vvVolDesc(const char* fn, size_t w, size_t h, size_t s, size_t f, float**, float**, float**);
    vvVolDesc(const char* fn, size_t w, size_t h, uint8_t* d);
    vvVolDesc(const vvVolDesc*, int=-1);
    virtual ~vvVolDesc();

    uint8_t* operator()(size_t x, size_t y, size_t slice);
    const uint8_t* operator()(size_t x, size_t y, size_t slice) const;
    uint8_t* operator()(size_t f, size_t x, size_t y, size_t slice);
    const uint8_t* operator()(size_t f, size_t x, size_t y, size_t slice) const;

    // Getters and setters:
    size_t getSliceBytes() const;
    size_t getFrameBytes() const;
    size_t getMovieBytes() const;
    size_t getSliceVoxels() const;
    size_t getFrameVoxels() const;
    size_t getMovieVoxels() const;
    virvo::aabb getBoundingBox() const;
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
    void   setDist(virvo::vec3f const& d);
    void   setRealRange(virvo::vec2 range);
    void   setRealRange(size_t channel, virvo::vec2 range);
    virvo::vec3f getSize() const;
    size_t getStoredFrames() const;
    float  getValueRange(size_t channel = 0) const;

    // Conversion routines:
    void   convertBPC(size_t, bool=false);
    void   convertChannels(size_t, int frame=-1, bool=false);
    void   deleteChannel(size_t, bool=false);
    void   bitShiftData(int, int frame=-1, bool=false);
    void   invert();
    void   convertRGB24toRGB8();
    void   flip(virvo::cartesian_axis< 3 > axis);
    void   rotate(virvo::cartesian_axis< 3 > axis, int dir);
    void   convertRGBPlanarToRGBInterleaved(int frame=-1);
    void   toggleEndianness(int frame=-1);
    void   crop(ssize_t x, ssize_t y, ssize_t z, ssize_t w, ssize_t h, ssize_t s);
    void   cropTimesteps(size_t start, size_t steps);
    void   resize(ssize_t w, ssize_t h, ssize_t s, InterpolationType, bool=false);
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
    void   makeUnsigned(int frame=-1);
    void   blend(vvVolDesc*, int, bool=false);
    void   swapChannels(size_t, size_t, bool=false);
    void   extractChannel(float[3], bool=false);
    bool   makeHeightField(size_t, int, bool=false);

    // Other routines:
    ErrorType merge(vvVolDesc*, vvVolDesc::MergeType);
    ErrorType mergeFrames(ssize_t slicesPerFrame=-1);
    void   addFrame(uint8_t*, DeleteType, int fd=-1);
    void   copyFrame(uint8_t*);
    void   removeSequence();
    void   makeHistogram(int, size_t, size_t, unsigned int*, int*, float, float);
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
    void   printVoxelData(int frame, ssize_t slice, ssize_t width=0, ssize_t height=0);
    void   printHistogram(int frame, size_t channel);
    void   trilinearInterpolation(size_t f, float, float, float, uint8_t*);
    void   drawBox(ssize_t p1x, ssize_t p1y, ssize_t p1z, ssize_t p2x, ssize_t p2y, ssize_t p2z, size_t chan, uint8_t* val);
    void   drawSphere(ssize_t p1x, ssize_t p1y, ssize_t p1z, ssize_t radius, size_t chan, uint8_t* val);
    void   drawLine(ssize_t p1x, ssize_t p1y, ssize_t p1z, ssize_t p2x, ssize_t p2y, ssize_t p2z, uint8_t*);
    void   drawBoundaries(uchar*, int=-1);
    size_t serializeAttributes(uint8_t* = NULL) const;
    void   deserializeAttributes(uint8_t*, size_t bufSize=SERIAL_ATTRIB_SIZE);
    void   setSliceData(uint8_t*, int=0, int=0);
    void   extractSliceData(int, virvo::cartesian_axis< 3 > axis, size_t slice, uint8_t*);
    void   makeSliceImage(int, virvo::cartesian_axis< 3 > axis, size_t slice, uint8_t*);
    void   getVolumeSize(virvo::cartesian_axis< 3 > axis, size_t&, size_t&, size_t&);
    void   deinterlace();
    void   findMinMax(size_t channel, float&, float&);
    int    findNumValue(int, float);
    int    findNumUsed(size_t channel);
    int    findNumTransparent(int);
    void   findDataBounds(ssize_t &x, ssize_t &y, ssize_t &z, ssize_t &w, ssize_t &h, ssize_t &s) const;
    void   calculateDistribution(int frame, size_t chan, float&, float&, float&);
    void   voxelStatistics(size_t frame, size_t c, ssize_t x, ssize_t y, ssize_t z, float&, float&);
    float  calculateMean(int);
    float  findClampValue(int, size_t channel, float);
    void   computeVolume(int, size_t, size_t, size_t);
    void   resizeEdgeMax(float);
    float  getChannelValue(int frame, size_t x, size_t y, size_t z, size_t chan);
    void   getLineHistData(int, int, int, int, int, int, std::vector< std::vector< float > >& resArray);
    void   setDefaultRealMinMax(size_t channel = 0);
    void   addGradient(size_t srcChan, GradientType);
    void   addVariance(size_t srcChan);
    void   deleteChannelNames();
    void   setChannelName(size_t, std::string const& name);
    std::string getChannelName(size_t) const;
    void updateFrame(int, uint8_t*, DeleteType);
    void updateHDRBins(size_t numValues, bool, bool, bool, BinningType, bool);
    int  findHDRBin(float);
    int  mapFloat2Int(float);
    void makeBinTexture(uint8_t* texture, size_t width);
    void computeTFTexture(size_t chan, size_t w, size_t h, size_t d, float* dest);
    void computeTFTexture(size_t w, size_t h, size_t d, float* dest); // tf for channel 0
    void makeLineTexture(DiagType, uchar, int, int, bool, std::vector< std::vector< float > > const& voxData, uint8_t*);
    void makeLineHistogram(size_t channel, int buckets, std::vector< std::vector< float > > const& data, int*);
    void computeMinMaxArrays(uint8_t *minArray, uchar *maxArray, ssize_t downsample, size_t channel=0, int frame=-1) const;
    virvo::vector< 3, ssize_t > voxelCoords(virvo::vec3f const& objCoords) const;
    virvo::vec3f objectCoords(virvo::vector< 3, ssize_t > const& voxCoords) const;

  private:
    char*  filename;                              ///< name of volume data file, including extension, excluding path ("" if undefined)
    size_t currentFrame;                          ///< current animation frame
    mutable vvSLList<uint8_t*> raw;               ///< pointer list to raw volume data - mutable because of Java style iterators
    std::vector<int> rawFrameNumber;           ///< frame numbers (if frames do not come in sequence)
    std::vector< std::string > channelNames;      ///< names of data channels

    void initialize();
    void setDefaults();
    void makeLineIntensDiag(size_t channel, std::vector< std::vector< float > > const& data, size_t numValues, int*);
    bool isChannelOn(size_t num, unsigned char);
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
