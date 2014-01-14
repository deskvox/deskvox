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

#include <iostream>
#include <iomanip>
#include <algorithm>    // required for std::sort function
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sstream>

#include <boost/detail/endian.hpp>

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

// Virvo:
#include "vvaabb.h"
#include "vvplatform.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"
#include "vvvecmath.h"
#include "vvclock.h"
#include "vvvoldesc.h"
#include "mem/swap.h"

#ifdef __sun
#define logf log
#endif

using namespace std;

const size_t vvVolDesc::DEFAULT_ICON_SIZE = 64;
const size_t vvVolDesc::NUM_HDR_BINS = 256;

//============================================================================
// Class vvVolDesc
//============================================================================

//----------------------------------------------------------------------------
/// Constructor.
vvVolDesc::vvVolDesc()
{
  vvDebugMsg::msg(2, "vvVolDesc::vvVolDesc(1)");
  initialize();
}

//----------------------------------------------------------------------------
/** Constructor with filename initialization.
 @param fn  volume file name
*/
vvVolDesc::vvVolDesc(const char* fn)
{
  vvDebugMsg::msg(2, "vvVolDesc::vvVolDesc(2)");
  initialize();
  setFilename(fn);
}

//----------------------------------------------------------------------------
/** Constructor with volume data initialization.
  The volume data will not be replicated and not be deleted upon deletion of
  vvVolDesc, so it _must_ be deleted by the caller!
 @param fn  volume file name (use "COVISE" if source is COVISE)
 @param w   width in pixels
 @param h   height in pixels
 @param s   number of volume slices
 @param f   number of animation frames
 @param b   number of bytes per channel
 @param m   number of channels
 @param d   pointer to pointer array of raw voxel data
*/
vvVolDesc::vvVolDesc(const char* fn, size_t w, size_t h, size_t s, size_t f, size_t b, size_t m,
uint8_t** d, vvVolDesc::DeleteType deleteType)
{
  uint8_t* data;

  vvDebugMsg::msg(2, "vvVolDesc::vvVolDesc(3)");
  initialize();
  setFilename(fn);
  vox[0] = w;
  vox[1] = h;
  vox[2] = s;
  bpc    = b;
  chan    = m;
  frames = f;

  if(d)
  {
    for (size_t i=0; i<f; ++i)
    {
      // Replicate data if necessary
      if(deleteType == ARRAY_DELETE)
      {
        addFrame(d[i], ARRAY_DELETE);
      }
      else
      {
        data = new uint8_t[getFrameBytes()];
        memcpy(data, d[i], getFrameBytes());
        addFrame(data, ARRAY_DELETE);
        if(deleteType == NORMAL_DELETE)
        {
          vvDebugMsg::msg(1, "vvVolDesc::vvVolDesc(3): NORMAL_DELETE does not make any sense for arrays");
          //delete d;
        }
      }
    }
    if (strcmp("COVISE", fn)==0)                  // convert data if source is COVISE
      convertCoviseToVirvo();
  }
}

//----------------------------------------------------------------------------
/** Constructor with volume data initialization
    in floating point format for density only data.
 @param fn     file name (use "COVISE" if source is COVISE)
 @param w,h,s  width, height, slices
 @param f      number of animation frames
 @param d      pointer array to raw voxel data (must be deleted by caller!)
*/
vvVolDesc::vvVolDesc(const char* fn, size_t w, size_t h, size_t s, size_t f, float** d)
{
  uint8_t* data;
  float frameMin, frameMax;                       // minimum and maximum value in one frame

  vvDebugMsg::msg(2, "vvVolDesc::vvVolDesc(4)");
  initialize();
  setFilename(fn);
  vox[0] = w;
  vox[1] = h;
  vox[2] = s;
  bpc    = 1;
  chan    = 1;
  frames = f;
  real[0] = VV_FLT_MAX;
  real[1] = -VV_FLT_MAX;

  for (size_t i=0; i<f; ++i)
  {
    vvToolshed::getMinMaxIgnore(d[i], getFrameVoxels(), VV_FLT_MAX, &frameMin, &frameMax);
    if (frameMin < real[0]) real[0] = frameMin;
    if (frameMax > real[1]) real[1] = frameMax;
  }
  for (size_t i=0; i<f; ++i)
  {
    data = new uint8_t[getFrameBytes()];
    vvToolshed::convertFloat2UCharClampZero(d[i], data, getFrameVoxels(),
      real[0], real[1], VV_FLT_MAX);
    addFrame(data, ARRAY_DELETE);
  }
  if (strcmp("COVISE", fn)==0)                    // convert data if source is COVISE
  {
    convertCoviseToVirvo();
  }
}

//----------------------------------------------------------------------------
/** Constructor with volume data initialization for one timestep
 in floating point format for RGB data
 @param fn     file name (use "COVISE" if source is COVISE)
 @param w,h,s  width, height, slices
 @param f      number of animation frames
 @param r,g,b  pointer arrays to raw voxel data for red, green, blue [0.0f..1.0f]
               (must be deleted by caller!)
*/
vvVolDesc::vvVolDesc(const char* fn, size_t w, size_t h, size_t s, size_t f, float** r,
float** g, float** b)
{
  uint8_t* data;
  float* c[3];
  size_t frameSize;                               // number of bytes per volume animation frame
  size_t compSize;                                // size of one color component

  vvDebugMsg::msg(2, "vvVolDesc::vvVolDesc(5)");
  initialize();
  vox[0] = w;
  vox[1] = h;
  vox[2] = s;
  bpc    = 1;
  chan    = 3;
  frames = f;
  frameSize = getFrameBytes();
  compSize  = getFrameVoxels();

  // Convert float to uchar:
  for (size_t k=0; k<f; ++k)
  {
    c[0] = r[k];
    c[1] = g[k];
    c[2] = b[k];
    data = new uint8_t[frameSize];
    for (size_t j=0; j<3; ++j)
      for (size_t i=0; i<compSize; ++i)
        data[i * 3 + j] = (uint8_t)(255.0f * c[j][i]);
    addFrame(data, ARRAY_DELETE);
  }
  if (strcmp("COVISE", fn)==0)                    // convert data if source is COVISE
  {
    convertCoviseToVirvo();
  }
}

//----------------------------------------------------------------------------
/** Constructor with volume data initialization for a 2D image in RGB format.
 @param fn   image file name
 @param w,h  width and height of image in pixels
 @param d    pointer to raw voxel data (must be deleted by caller!)
*/
vvVolDesc::vvVolDesc(const char* fn, size_t w, size_t h, uint8_t* d)
{
  vvDebugMsg::msg(2, "vvVolDesc::vvVolDesc(6)");
  initialize();
  setFilename(fn);
  vox[0] = w;
  vox[1] = h;
  vox[2] = 1;
  bpc    = 1;
  chan    = 3;
  frames = 1;
  uint8_t* data = new uint8_t[getFrameBytes()];
  memcpy(data, d, getFrameBytes());
  addFrame(data, ARRAY_DELETE);
}

//----------------------------------------------------------------------------
/** Copy constructor.
 Copies all vvVolDesc data including transfer functions and raw voxel data.
 @param v  source volume description
 @param f  frame index to copy (0 for first frame, -1 for all frames, -2 for no copying of raw data)
*/
vvVolDesc::vvVolDesc(const vvVolDesc* v, int f)
{
  vvDebugMsg::msg(2, "vvVolDesc::vvVolDesc(7)");
  initialize();
  setFilename(v->filename);
  for (size_t i=0; i<3; ++i)
  {
    vox[i]  = v->vox[i];
    dist[i] = v->dist[i];
  }
  for (size_t i=0; i<2; ++i)
  {
    real[i] = v->real[i];
  }
  dt     = v->dt;
  bpc    = v->bpc;
  chan   = v->chan;
  frames = 0;
  currentFrame = (f==-1) ? v->currentFrame : 0;

  // Copy icon:
  iconSize = v->iconSize;
  if (iconSize > 0)
  {
    iconData = new uint8_t[iconSize * iconSize * ICON_BPP];
    memcpy(iconData, v->iconData, iconSize * iconSize * ICON_BPP);
  }

  // Copy transfer functions:
  tf.copy(&tf._widgets, &v->tf._widgets);

  if (f==-1)
  {
    for (size_t i=0; i<v->frames; ++i)
    {
      copyFrame(v->getRaw(i));
      ++frames;
    }
  }
  else if (f!=-2)
  {
    copyFrame(v->getRaw(size_t(f)));
    ++frames;
  }
}

//----------------------------------------------------------------------------
/// Destructor
vvVolDesc::~vvVolDesc()
{
  vvDebugMsg::msg(2, "vvVolDesc::~vvVolDesc()");
  removeSequence();
  delete[] filename;
  delete[] iconData;
  delete[] _hdrBinLimits;
}

uint8_t* vvVolDesc::operator()(size_t x, size_t y, size_t slice)
{
  uint8_t* raw = getRaw(-1);
  raw += slice * getSliceBytes();
  return &raw[(vox[0] * y + x) * getBPV()];
}

const uint8_t* vvVolDesc::operator()(size_t x, size_t y, size_t slice) const
{
  const uint8_t* raw = getRaw(-1);
  raw += slice * getSliceBytes();
  return &raw[(vox[0] * y + x) * getBPV()];
}

uint8_t* vvVolDesc::operator()(size_t f, size_t x, size_t y, size_t slice)
{
  uint8_t* raw = getRaw(f);
  raw += slice * getSliceBytes();
  return &raw[(vox[0] * y + x) * getBPV()];
}

const uint8_t* vvVolDesc::operator()(size_t f, size_t x, size_t y, size_t slice) const
{
  uint8_t* raw = getRaw(f);
  raw += slice * getSliceBytes();
  return &raw[(vox[0] * y + x) * getBPV()];
}

//----------------------------------------------------------------------------
/** Initialization of class attributes.
  May only be called by constructors!
*/
void vvVolDesc::initialize()
{
  vvDebugMsg::msg(2, "vvVolDesc::initialize()");
  setDefaults();
  removeSequence();
  currentFrame = 0;
  filename = NULL;
  _mask = NULL;
  _radius = 0;
  for (std::vector<vvTFWidget*>::const_iterator it = tf._widgets.begin();
       it != tf._widgets.end(); ++it)
  {
    delete *it;
  }
  tf._widgets.clear();
  iconSize = 0;
  _scale = 1.0f;
  _hdrBinLimits = new float[NUM_HDR_BINS];
  _binning = LINEAR;
  _transOp = false;
  iconData = NULL;
}

//----------------------------------------------------------------------------
/// Set default values for all serializable attributes.
void vvVolDesc::setDefaults()
{
  vvDebugMsg::msg(2, "vvVolDesc::setDefaults()");

  vox[0] = vox[1] = vox[2] = frames = 0;
  frames = 0;
  bpc = 1;
  chan = 0;
  dist[0] = dist[1] = dist[2] = 1.0f;
  dt = 1.0f;
  real[0] = 0.0f;
  real[1] = 1.0f;
  pos.zero();
}

//----------------------------------------------------------------------------
/// Remove all frames of animation sequence.
void vvVolDesc::removeSequence()
{
  vvDebugMsg::msg(2, "vvVolDesc::removeSequence()");
  if (raw.isEmpty()) return;
  raw.removeAll();
  deleteChannelNames();
}

//----------------------------------------------------------------------------
/** Delete all channel names.
 */
void vvVolDesc::deleteChannelNames()
{
  channelNames.clear();
}

//----------------------------------------------------------------------------
/** Set a channel name.
  @param channel channel index to set name of (0=first channel)
  @param name new channel name; name will be copied: caller has to delete name.
         NULL may be passed as a name.
*/
void vvVolDesc::setChannelName(size_t channel, std::string const& name)
{
  vvDebugMsg::msg(2, "vvVolDesc::setChannelName()");

  if (channel >= channelNames.size())
  {
    channelNames.resize(channel + 1);
  }

  channelNames[channel] = name;
}

//----------------------------------------------------------------------------
/** @return a channel name.
  @param channel channel index to get name of (0=first channel)
*/
std::string vvVolDesc::getChannelName(size_t channel) const
{
  vvDebugMsg::msg(2, "vvVolDesc::getChannelName()");
  assert(channel < channelNames.size());
  return channelNames[channel];
}

//----------------------------------------------------------------------------
/** Get slice size.
 @return slice size in number of bytes
*/
size_t vvVolDesc::getSliceBytes() const
{
  return vox[0] * vox[1] * getBPV();
}

//----------------------------------------------------------------------------
/** Get frame size.
 @return frame size in number of bytes
*/
size_t vvVolDesc::getFrameBytes() const
{
  return vox[0] * vox[1] * vox[2] * getBPV();
}

//----------------------------------------------------------------------------
/** Get movie size.
 @return movie size in bytes (movie = timely sequence of all volume frames)
*/
size_t vvVolDesc::getMovieBytes() const
{
  return vox[0] * vox[1] * vox[2] * frames * getBPV();
}

//----------------------------------------------------------------------------
/// Get number of voxels in a slice.
size_t vvVolDesc::getSliceVoxels() const
{
  return vox[0] * vox[1];
}

//----------------------------------------------------------------------------
/// Get number of voxels in a frame.
size_t vvVolDesc::getFrameVoxels() const
{
  return vox[0] * vox[1] * vox[2];
}

//----------------------------------------------------------------------------
/// Get number of voxels in the volume movie.
size_t vvVolDesc::getMovieVoxels() const
{
  return vox[0] * vox[1] * vox[2] * frames;
}

//----------------------------------------------------------------------------
/// Get bounding box of the volume.
void vvVolDesc::getBoundingBox(vvAABB& aabb) const
{
  const vvVector3 size2 = getSize() * 0.5f;
  aabb = vvAABB(pos - size2, pos + size2);
}

//----------------------------------------------------------------------------
/** Merge two volume datasets into one file.
 The data will be moved to the target volume.<BR>
 No problems occur if either one of the sequences is empty.<BR>
 @param src new volume sequence. Will end up with no volume data.
 @param mtype merge type
 @return OK if successful
*/
vvVolDesc::ErrorType vvVolDesc::merge(vvVolDesc* src, vvVolDesc::MergeType mtype)
{
  uint8_t* newRaw;
  uint8_t* rd;

  vvDebugMsg::msg(2, "vvVolDesc::merge()");
  if (src->frames==0) return OK;                  // is source src empty?
                                                  // are data types the same?
  if ((bpc != src->bpc) && frames != 0) return TYPE_ERROR;

  // If target VD empty: create a copy:
  if (frames==0)
  {
    // Copy all volume data to this volume:
    vox[0] = src->vox[0];
    vox[1] = src->vox[1];
    vox[2] = src->vox[2];
    frames = src->frames;
    bpc    = src->bpc;
    chan   = src->chan;
    dt     = src->dt;
    raw.merge(&src->raw);
    for (size_t i=0; i<3; ++i) dist[i] = src->dist[i];
    for (size_t i=0; i<2; ++i)
    {
      real[i] = src->real[i];
    }
    for (size_t i=0; i<chan; ++i) setChannelName(i, src->channelNames[i]);
    pos = src->pos;
    currentFrame = src->currentFrame;
    tf.copy(&tf._widgets, &src->tf._widgets);

    // Delete sequence information from source:
    src->bpc = src->chan = src->vox[0] = src->vox[1] = src->vox[2] = src->frames = src->currentFrame = 0;
    return OK;
  }

  if (mtype==vvVolDesc::VV_MERGE_CHAN2VOL)
  {
    // If dimensions and time steps match: add channels
    if (vox[0]==src->vox[0] && vox[1]==src->vox[1] &&
      vox[2]==src->vox[2] && bpc==src->bpc && frames==src->frames)
    {
      for (size_t f=0; f<frames; ++f)
      {
        raw.makeCurrent(f);
        rd = raw.getData();
        uint8_t* srcRD = src->getRaw(f);
        newRaw = new uint8_t[getFrameBytes() + src->getFrameBytes()];
        for (size_t i=0; i<getFrameVoxels(); ++i)
        {
          memcpy(newRaw + i * bpc * (chan+src->chan), rd + i * getBPV(), getBPV());
          memcpy(newRaw + i * bpc * (chan+src->chan) + getBPV(), srcRD + i * src->getBPV(), src->getBPV());
        }
        raw.remove();
        if (f==0) raw.insertBefore(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
        else raw.insertAfter(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
      }
      for (size_t i=0; i<src->chan; ++i) setChannelName((chan+i), src->channelNames[i]);
      chan += src->chan;                          // update target channel number
      src->removeSequence();                      // delete copied frames from source
      return OK;
    }
    else
    {
      cerr << "Channels merge error: volume parameters do not match." << endl;
      return TYPE_ERROR;
    }
  }
  else if (mtype==vvVolDesc::VV_MERGE_VOL2ANIM)
  {
    // If dimensions and channels match, and there is more than one slice: treat as time steps
    if (vox[0]==src->vox[0] && vox[1]==src->vox[1] && vox[2]==src->vox[2] &&
      bpc==src->bpc && chan==src->chan)
    {
      // Append all volume time steps to target volume:
      raw.merge(&src->raw);
      frames = raw.count();

      // Delete sequence information from src:
      src->bpc = src->chan = src->vox[0] = src->vox[1] = src->vox[2] = src->frames = src->currentFrame = 0;
      src->deleteChannelNames();
      return OK;
    }
    else
    {
      cerr << "Volumes merge error: volume parameters do not match." << endl;
      return TYPE_ERROR;
    }
  }
  else if (mtype==vvVolDesc::VV_MERGE_SLABS2VOL)
  {
    // If slice dimensions and animation length match: append volumes in each time step
    if (vox[0]==src->vox[0] && vox[1]==src->vox[1] && frames==src->frames)
    {
      // Append all slices of each animation step to target volume:
      for (size_t f=0; f<frames; ++f)
      {
        raw.makeCurrent(f);
        rd = raw.getData();
        newRaw = new uint8_t[getFrameBytes() + src->getFrameBytes()];
        memcpy(newRaw, rd, getFrameBytes());      // copy current frame to new raw data array
                                                  // copy source frame to new raw data array
        memcpy(newRaw + getFrameBytes(), src->getRaw(f), src->getFrameBytes());
        raw.remove();
        if (f==0) raw.insertBefore(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
        else raw.insertAfter(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
      }
      vox[2] += src->vox[2];                      // update target slice number
      src->removeSequence();                      // delete copied frames from source
      return OK;
    }
    else
    {
      cerr << "Slices merge error: volume parameters do not match." << endl;
      return TYPE_ERROR;
    }
  }
  return TYPE_ERROR;                              // more sophisticated merging methods can be implemented later on
}

//----------------------------------------------------------------------------
/** Merge all the frames to one volume
 This will typically be used for generating a volume from a sequence of files
 @return OK if successful
*/
vvVolDesc::ErrorType vvVolDesc::mergeFrames()
{
  uint8_t *newRaw = new uint8_t[getFrameBytes() * frames];
  for (size_t f=0; f<frames; f++)
  {
    int fni = rawFrameNumber[f]-1;
    size_t fn;
    if(fni<0)
      fn = f;
    else
      fn = size_t(fni);

    if(fn >= frames)
    {
       fn = frames-1;
       fprintf(stderr," please read all frames, not enough space for all frames\n");
    }
    memcpy(newRaw + getFrameBytes()*fn, getRaw(f), getFrameBytes());
  }
  removeSequence();
  addFrame(newRaw, ARRAY_DELETE);
  vox[2] = frames;
  frames = 1;

  return OK;
}

//----------------------------------------------------------------------------
/** Legacy function, getRaw(-1) returns data for current frame
  @param frame  has default value -1
*/
uint8_t* vvVolDesc::getRaw(int) const
{
  return getRaw(currentFrame);
}

//----------------------------------------------------------------------------
/** Returns a pointer to the raw data of a specific frame.
  @param frame  index of desired frame (0 for first frame) if frame does not
                exist, NULL will be returned
*/
uint8_t* vvVolDesc::getRaw(size_t frame) const
{
  if (frame>=frames) return NULL;     // frame does not exist
  raw.makeCurrent(frame);
  return raw.getData();
}

//----------------------------------------------------------------------------
/** Adds a new frame to the animation sequence. The data has to be in
    the appropriate format, according to the vvVolDesc::bpv setting.
    The data itself is not copied to the sequence.
    It has to be deleted by the caller if deleteData is false,
    or is deleted automatically if deleteData is true.
    <BR>
    The frames variable is not adjusted, this must be done separately.
    <BR>
    The data bytes are stored in the following order:
    Starting point is the top left voxel on the
    front slice. For each voxel, bpv bytes are stored with
the most significant byte first. The voxels are stored
in English writing order as in a book (first left to right,
then top to bottom, then front to back).
After the front slice is stored, the second one is stored in the
same order of voxels. Last stored is the right bottom voxel of
the back slice.
@param ptr          pointer to raw data
@param deleteData   data deletion type: delete or don't delete when not used anymore
*/
void vvVolDesc::addFrame(uint8_t* ptr, DeleteType deleteData,int fn)
{
  switch(deleteData)
  {
    case NO_DELETE:     raw.append(ptr, vvSLNode<uint8_t*>::NO_DELETE); break;
    case NORMAL_DELETE: raw.append(ptr, vvSLNode<uint8_t*>::NORMAL_DELETE); break;
    case ARRAY_DELETE:  raw.append(ptr, vvSLNode<uint8_t*>::ARRAY_DELETE); break;
    default: assert(0); break;
  }
  rawFrameNumber.push_back(fn);

  // Make sure channel names exist:
  if (channelNames.size() == 0)
  {
    channelNames.resize(chan);
  }
}

//----------------------------------------------------------------------------
/// Return the number of frames actually stored.
size_t vvVolDesc::getStoredFrames() const
{
  return raw.count();
}

//----------------------------------------------------------------------------
/** Copies frame data from memory and adds them as a new frame
    to the animation sequence.
  @param ptr  pointer to raw source data, _must_ be deleted by the caller!
*/
void vvVolDesc::copyFrame(uint8_t* ptr)
{
  uint8_t* newData;

  vvDebugMsg::msg(3, "vvVolDesc::copyFrame()");
  newData = new uint8_t[getFrameBytes()];
  memcpy(newData, ptr, getFrameBytes());
  raw.append(newData, vvSLNode<uint8_t*>::ARRAY_DELETE);

  // Make sure channel names exist:
  if (channelNames.size() == 0)
  {
    channelNames.resize(chan);
  }
}

//----------------------------------------------------------------------------
/** Updates the volume data of a frame. The data format needs to stay
the same.
  @param frame frame ID (0=first)
  @param newData pointer to raw source data.
  @param deleteData   data deletion type: delete or don't delete when not used anymore
*/
void vvVolDesc::updateFrame(int frame, uint8_t* newData, DeleteType deleteData)
{
  vvDebugMsg::msg(3, "vvVolDesc::updateFrame()");
  raw.makeCurrent(frame);
  raw.remove();
  switch(deleteData)
  {
    case NO_DELETE:     raw.insertAfter(newData, vvSLNode<uint8_t*>::NO_DELETE); break;
    case NORMAL_DELETE: raw.insertAfter(newData, vvSLNode<uint8_t*>::NORMAL_DELETE); break;
    case ARRAY_DELETE:  raw.insertAfter(newData, vvSLNode<uint8_t*>::ARRAY_DELETE); break;
    default: assert(0); break;
  }
}

//----------------------------------------------------------------------------
/** Normalize the values of a histogram to the range [0..1].
  @param buckets number of entries in arrays
  @param count   absolute histogram values
  @param normalized pointer to _allocated_ array with buckets entries
  @param type    type of normalization
  @return normalized values in 'normalized'
*/
void vvVolDesc::normalizeHistogram(int buckets, int* count, float* normalized, NormalizationType type)
{
  int max = 0;                                    // maximum counter value
  int i;

  vvDebugMsg::msg(2, "vvVolDesc::normalizeHistogram()");
  assert(count);

  // Find maximum counter value:
  for (i=0; i<buckets; ++i)
    max = ts_max(max, count[i]);

  if (max == 1) type = VV_LINEAR;

  // Normalize counter values:
  for (i=0; i<buckets; ++i)
  {
    if (count[i] < 1) normalized[i] = 0.0f;
    else if (type==VV_LOGARITHMIC)
    {
      normalized[i] = logf(float(count[i])) / logf(float(max));
    }
    else normalized[i] = float(count[i]) / float(max);
  }
}

//----------------------------------------------------------------------------
/** Generate voxel value histogram array.
  Counts:
  <ul>
    <li>number of data value occurrences for 8 or 16 bit per voxel data</li>
    <li>number of occurrences of each color component for RGB data</li>
    <li>number of density value occurrences for 32 bit per voxel data</li>
  </ul>
  @param frame   frame to generate histogram for (0 is first frame, -1 for all frames)
  @param chan1   first channel to create histogram for (0 is first channel of data set)
  @param numChan number of channels to create histogram for (determines dimensionality)
  @param buckets number of counters to use for histogram computation in each dimension (expects array int[numChan])
  @param count   _allocated_ array with 'buckets[0] * buckets[1] * ...' entries of type int.
  @param min,max data range for which histogram is to be created. Use 0..1 for integer data types.
  @return histogram values in 'count'
*/
void vvVolDesc::makeHistogram(int frame, size_t chan1, size_t numChan, int* buckets, int* count, float min, float max)
{
  uint8_t* raw;                                   // raw voxel data
  float* voxVal;                                  // voxel values
  float* valPerBucket;                            // scalar data values per bucket
  size_t c, m;                                    // counters
  size_t srcIndex;                                // index into raw volume data
  size_t dstIndex;                                // index into histogram array
  size_t factor;                                  // multiplication factor for dstIndex
  size_t numVoxels;                               // number of voxels per frame
  int* bucket;                                    // bucket ID
  size_t srcChan;                                 // channel index in data set
  int totalBuckets;                               // total number of buckets

  vvDebugMsg::msg(2, "vvVolDesc::makeHistogram()");

  totalBuckets = 1;
  for (c=0; c<numChan; ++c)
  {
    totalBuckets *= buckets[c];
  }
  memset(count, 0, totalBuckets * sizeof(int));   // initialize counter array

  voxVal = new float[numChan];
  bucket = new int[numChan];
  valPerBucket = new float[numChan];
  numVoxels = getFrameVoxels();
  for (c=0; c<numChan; ++c)
  {
    if (bpc==4) valPerBucket[c] = (max-min) / float(buckets[c]);
    else valPerBucket[c] = getValueRange() / float(buckets[c]);
  }
  for (size_t f=0; f<frames; ++f)
  {
    if (frame != -1 && frame != f)
      continue; // only compute histogram for a specific frame

    raw = getRaw(f);
    for (size_t i=0; i<numVoxels; ++i)                   // count each voxel value
    {
      srcIndex = i * getBPV();
      dstIndex = 0;
      for (size_t c=0; c<numChan; ++c)
      {
        srcChan = ts_min(c, chan);
        switch (bpc)
        {
          case 1:
            voxVal[c] = float(raw[srcIndex + chan1 + srcChan]);
            break;
          case 2:
            voxVal[c] = float(int(raw[srcIndex + 2 * (chan1 + srcChan)] << 8) | int(raw[srcIndex + 2 * (chan1 + srcChan) + 1]));
            break;
          case 4:
            voxVal[c] = *((float*)(raw + srcIndex + 4 * (chan1 + srcChan)));
            break;
          default: assert(0); break;
        }

        bucket[c] = int(float(voxVal[c] - ((bpc==4) ? min : 0)) / valPerBucket[c]);
        bucket[c] = ts_clamp(bucket[c], 0, buckets[c]-1);
        factor = 1;
        for (m=0; m<c; ++m)
        {
          factor *= buckets[m];
        }
        dstIndex += bucket[c] * factor;
      }
      ++count[dstIndex];
    }
  }

  delete[] valPerBucket;
  delete[] bucket;
  delete[] voxVal;
}

//----------------------------------------------------------------------------
/** Generates RGB texture values for the histogram.
 Texture values are returned as 3 bytes per texel, bottom to top,
 ordered RGBRGBRGB...
 The y axis is logarithmic.
  @param frame           animation frame to make histogram texture for (-1 for all frames)
  @param chan1           first channel to make texture for (0 for first channel in data set)
  @param numChan         number of consecutive channels to calculate histogram for
  @param size            array of edge lengths for texture [texels]
  @param data            pointer to _pre-allocated_ memory space providing
                         twidth * theight * 4 bytes
  @param color           color for histogram foreground (background is transparent)
  @param min,max         data range for which histogram is to be created. Use 0..1 for integer data types.
*/
void vvVolDesc::makeHistogramTexture(int frame, size_t chan1, size_t numChan, size_t* size, uint8_t* data,
  NormalizationType ntype, vvColor* color, float min, float max)
{
  const size_t BPT = 4;                           // bytes per texel
  float* hist;                                    // histogram values (float)
  int*   count;                                   // histogram values (integer)
  size_t x,y;                                     // texel index
  size_t barHeight;                               // histogram bar height [texels]
  size_t c;                                       // color component counter
  size_t texIndex;                                // index in TF texture
  size_t numValues;
  int    histIndex;
  int*   buckets;                                 // histogram dimensions
  int*   index;                                   // index in histogram array

  vvDebugMsg::msg(2, "vvVolDesc::makeHistogramTexture()");
  assert(data);
  buckets = new int[numChan];
  index = new int[numChan];
  numValues = 1;
  for (c=0; c<numChan; ++c)
  {
    buckets[c] = size[c];
    numValues *= buckets[c];
  }
  count = new int[numValues];
  hist = new float[numValues];
  makeHistogram(frame, chan1, numChan, buckets, count, min, max);
  normalizeHistogram(numValues, count, hist, ntype);

  // Draw histogram:
  if (numChan==1)                                 // 1 channel: draw bar chart
  {
    // Fill histogram texture with background values:
    memset(data, 0, size[0] * size[1] * BPT);
    for (x=0; x<size[0]; ++x)
    {
      index[0] = int(float(x) / float(size[0]) * float(buckets[0]));
      // Find height of histogram bar:
      barHeight = size_t(hist[index[0]] * float(size[1]-1));
      for (y=0; y<barHeight; ++y)
      {
        texIndex = BPT * (x + y * size[0]);
        for (c=0; c<3; ++c)
        {
          data[texIndex + c] = uchar((*color)[c] * 255.0f);        // set foreground color
        }
        data[texIndex + 3] = 255;                 // set alpha
      }
    }
  }
  else if (numChan==2)                            // >1 channels: draw scatterplot
  {
    for (y=0; y<size[1]; ++y)
    {
      for (x=0; x<size[0]; ++x)
      {
        index[0] = int(float(x) / float(size[0]) * float(buckets[0]));
        index[1] = int(float(y) / float(size[1]) * float(buckets[1]));
        texIndex = BPT * (x + y * size[0]);
        histIndex = index[0] + index[1] * buckets[0];
        for (c=0; c<3; ++c)
        {
                                                  // set foreground color
          data[texIndex + c] = int(hist[histIndex] * 255.0f);
        }
        data[texIndex + 3] = 255;                 // set alpha
      }
    }
  }
  delete[] hist;
  delete[] count;
  delete[] index;
  delete[] buckets;
}

/** Wrapper around tf.computeTFTexture.
*/
void vvVolDesc::computeTFTexture(int w, int h, int d, float* dest)
{
  const int RGBA = 4;
  float dataVal;
  int i, linearBin;

  if (this->chan == 2)
     tf.computeTFTexture(w, h, d, dest, real[0], real[1], 0.0f, 1.0f); //TODO substitute fixed values!
  else
     //default: act as 1D
     tf.computeTFTexture(w, h, d, dest, real[0], real[1]);

  // convert opacity TF if hdr mode:
  if (_binning!=LINEAR && !_transOp)
  {
     float* tmpOp = new float[w*h*d*RGBA];
     memcpy(tmpOp, dest, w * h * d * RGBA * sizeof(float));
     for (i=0; i<w; ++i) // go through all bins and non-linearize them
     {
        dataVal = _hdrBinLimits[i];
        linearBin = int((dataVal - real[0]) / (real[1] - real[0]) * float(w));
        linearBin = ts_clamp(linearBin, 0, w-1);
        dest[i*RGBA+3] = tmpOp[linearBin*RGBA+3];
     }
     delete[] tmpOp;
  }
}

/**
  This method creates a texture for the given voxeldata.
  @param type: should it be a histogram or intensity diagram
  @param selChannel: only the selected channels should be displayed
  @param twidth: texture width
  @param theight: texture height
  @param alpha: should texture contain an alpha channel
  @param voxData: voxeldata
  @param texData: the created texture data
*/
void vvVolDesc::makeLineTexture(DiagType type, unsigned char selChannel, int twidth, int theight, bool alpha,
  std::vector< std::vector< float > > const& voxData, uint8_t* texData)
{
  int x, y, c;
  int bpt;
  int* data;
  float *normalized=NULL;
  unsigned char bg[4]    = {255,255,255,255};
  unsigned char fg[4][4] =
  {
    {
      255,  0,  0,255
    },
    {  0,255,  0,255},
    {  0,  0,255,255},
    {  0,  0,  0,255}
  };
  int prevY, currY;
  unsigned char* tex;
  unsigned int color;

  assert(texData != 0);

  data = new int[twidth];
  if (type == HISTOGRAM) normalized = new float[twidth];

  if (alpha) bpt = 4;
  else bpt = 3;

  // initialize texture
  tex = texData;
  for (y = 0; y < theight; ++y)
    for (x = 0; x < twidth; ++x)
      for (c = 0; c < bpt; ++c)
      {
        *tex = bg[c];
        ++tex;
      }

  for (size_t i = 0; i < chan; ++i)
  {
    if (isChannelOn(i, selChannel))
    {
      if (type == HISTOGRAM)
      {
        makeLineHistogram(i, twidth, voxData, data);
        normalizeHistogram(twidth, data, normalized, VV_LOGARITHMIC);
      }
      else
      {
        makeLineIntensDiag(i, voxData, twidth, data);
      }

      // get color for current channel
      color = ((unsigned int) fg[i][0]) << 24;
      color = color | (((unsigned int) fg[i][1]) << 16);
      color = color | (((unsigned int) fg[i][2]) << 8);
      color = color | ((unsigned int) fg[i][3]);

      if (type == HISTOGRAM)
        prevY = (int)(normalized[0] * (theight-1));
      else
        prevY = ts_clamp(data[0], 0, theight-1);

      for (x = 1; x < twidth; ++x)
      {
        if (type == HISTOGRAM) currY = (int)(normalized[x] * (theight-1));
        else currY = ts_clamp(data[x], 0, theight-1);

        vvToolshed::draw2DLine(x-1, prevY, x, currY, color, texData, bpt, twidth, theight);
        prevY = currY;
      }
    }
  }

  delete[] data;
  if (type == HISTOGRAM) delete[] normalized;
}

//----------------------------------------------------------------------------
bool vvVolDesc::isChannelOn(size_t num, unsigned char selected)
{
  switch (num)
  {
    case 0:
      if ((CHANNEL_R & selected) != 0) return true;
      else return false;
    case 1:
      if ((CHANNEL_G & selected) != 0) return true;
      else return false;
    case 2:
      if ((CHANNEL_B & selected) != 0) return true;
      else return false;
    case 3:
      if ((CHANNEL_A & selected) != 0) return true;
      else return false;
    default: return false;
  }
}

void vvVolDesc::makeLineHistogram(size_t channel, int buckets, std::vector< std::vector< float > > const& data, int* count)
{
  size_t numVoxels;
  int bucket;
  float valPerBucket;
  float voxVal;

  memset(count, 0, buckets * sizeof(int));

  numVoxels = data.size();
  valPerBucket = getValueRange() / float(buckets);

  for (size_t i = 0; i < numVoxels; ++i)
  {
    voxVal = data[i][channel];
    bucket = int(voxVal / valPerBucket);
    bucket = ts_clamp(bucket, 0, buckets-1);
    ++count[bucket];
  }
}

void vvVolDesc::makeLineIntensDiag(size_t channel, std::vector< std::vector< float > > const& data, size_t numValues, int* values)
{
  float step;

  step = (float) data.size() / (float) numValues;

  memset(values, 0, numValues * sizeof(int));

  for (size_t i = 0; i < numValues; ++i)
  {
    size_t index = (size_t) floor((float) i * step);
    values[i] = (int) data[index][channel];
  }
}

//----------------------------------------------------------------------------
/** Create ASCII files with the histogram. For each channel
  a separate histogram file will be created.
  @param overwrite true to overwrite existing files
*/
void vvVolDesc::createHistogramFiles(bool overwrite)
{
  FILE* fp;
  int* hist;
  int buckets[1];

  char* basePath = new char[strlen(filename) + 1];
  char* fileName = new char[strlen(filename) + 15];
  vvToolshed::extractBasePath(basePath, filename);

  buckets[0] = int(getValueRange());
  hist = new int[buckets[0]];

  for (size_t m=0; m<chan; ++m)
  {
    if (chan > 1)
    {
      sprintf(fileName, "%s-hist-ch%02d.txt", basePath, static_cast<int32_t>(m));
    }
    else
    {
      sprintf(fileName, "%s-hist.txt", basePath);
    }

    // Compute histogram:
    makeHistogram(-1, m, 1, buckets, hist, real[0], real[1]);

    // Check if file exists:
    if (!overwrite && vvToolshed::isFile(fileName))
    {
      cerr << "Error - file exists: " << fileName << endl;
      continue;
    }

    // Write histogram values to file:
    if ( (fp = fopen(fileName, "wt")) == NULL)
    {
      cerr << "Error - cannot open: " << fileName << endl;
      continue;
    }
    for (int i=0; i<buckets[0]; ++i)
    {
      fprintf(fp, "%d\n", hist[i]);
    }
    fclose(fp);
  }
  delete[] hist;
  delete[] fileName;
  delete[] basePath;
}

//----------------------------------------------------------------------------
/** Set file name.
  @param fn  file name including path (e.g. "c:\data\volumes\testvol.xvf")
*/
void vvVolDesc::setFilename(const char* fn)
{
  delete[] filename;
  if (fn==NULL) filename = NULL;
  else
  {
    filename = new char[strlen(fn) + 1];
    strcpy(filename, fn);
  }
  return;
}

//----------------------------------------------------------------------------
/** Get file name.
  @return pointer to file name. Don't delete this pointer.
*/
const char* vvVolDesc::getFilename() const
{
  return filename;
}

//----------------------------------------------------------------------------
/** Set current frame.
  @param f  current frame
*/
void vvVolDesc::setCurrentFrame(size_t f)
{
  if (f<frames) currentFrame = f;
}

//----------------------------------------------------------------------------
/// Get current frame.
size_t vvVolDesc::getCurrentFrame() const
{
  return currentFrame;
}

//----------------------------------------------------------------------------
/// Get number of bytes per voxel.
size_t vvVolDesc::getBPV() const
{
  return bpc * chan;
}

//----------------------------------------------------------------------------
/** Get range of values in each channel. Depends only on setting of bpc.
    Useful for creating histograms.
 @return the range of values in each channel; returns 0.0f on error.
*/
float vvVolDesc::getValueRange() const
{
  switch(bpc)
  {
    case 1: return 256.0f;
    case 2: return 65536.0f;
    case 4: return real[1] - real[0];
    default: assert(0); return 0.0f;
  }
}

//----------------------------------------------------------------------------
/** Converts the number of bytes per channel.
  The strategy depends on the number of bytes per channel both in the source
  and in the destination volume:
  <PRE>
  Source  Destination  Strategy
  -----------------------------
     1       2         Shift source value left by 8 bit
     1       4         Convert to range [0..1]
     2       1         Shift source value right by 8 bit
     2       4         Convert to range [0..1]
     4       1         Convert range between realMin/Max to 8 bit
4       2         Convert range between realMin/Max to 16 bit
</PRE>
Each channel is converted.

@param newBPC  new number of bytes per channel
@param verbose true = print progress info
*/
void vvVolDesc::convertBPC(size_t newBPC, bool verbose)
{
  uint8_t* newRaw;
  uint8_t* rd;
  uint8_t* src;
  uint8_t* dst;
  float val;
  size_t newSliceSize;

  vvDebugMsg::msg(2, "vvVolDesc::convertBPC()");

  // Verify input parameters:
  if (bpc==newBPC) return;                        // this was easy!
  assert(newBPC==1 || newBPC==2 || newBPC==4);

  newSliceSize = vox[0] * vox[1] * newBPC * chan;
  if (verbose) vvToolshed::initProgress(vox[2] * frames);

  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    newRaw = new uchar[newSliceSize * vox[2]];
    src = rd;
    dst = newRaw;
    for (ssize_t z=0; z<vox[2]; ++z)
    {
      for (ssize_t y=0; y<vox[1]; ++y)
      {
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          for (size_t c=0; c<chan; ++c)
          {
            // Perform actual conversion:
            switch (bpc)                          // switch by source voxel type
            {
              case 1:                             // 8 bit source
                switch (newBPC)                   // switch by destination voxel type
                {
                  case 2:
                    virvo::serialization::write16(dst, src[0]);
                    break;
                  case 4:
                    *((float*)dst) = src[0] / 255.0f;
                    break;
                }
                break;
              case 2:                             // 16 bit source
                switch (newBPC)                   // switch by destination voxel type
                {
                  case 1:
                    *dst = src[0];
                    break;
                  case 4:
                    *((float*)dst) = (src[0] * 256.0f + src[1]) / 65535.0f;
                    break;
                }
                break;
              case 4:                             // float source
                val = ts_clamp(*((float*)src), real[0], real[1]);
                switch (newBPC)                   // switch by destination voxel type
                {
                  case 1:
                    *dst = uchar((val - real[0]) / (real[1] - real[0]) * 255.0f);
                    break;
                  case 2:
                    virvo::serialization::write16(dst, uint16_t((val - real[0]) / (real[1] - real[0]) * 65535.0f));
                    break;
                }
                break;
            }
            src += bpc;
            dst += newBPC;
          }
        }
      }
      if (verbose) vvToolshed::printProgress(z + vox[2] * f);
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
    raw.next();
  }
  bpc = newBPC;
}

//----------------------------------------------------------------------------
/** Convert the number of channels. If newChan > chan: pad with 0's,
  else delete starting with highest channel.
  @param newChan new number of channels
  @param verbose true = print progress info
*/
void vvVolDesc::convertChannels(size_t newChan, int frame, bool verbose)
{
  uint8_t* newRaw;
  uint8_t* rd;
  uint8_t* src;
  uint8_t* dst;
  size_t newSliceSize;

  vvDebugMsg::msg(2, "vvVolDesc::convertChannels()");

  if (chan==newChan) return;                      // this was easy!
  assert(newChan>0);                              // ignore invalid values

  size_t startFrame = 0;
  size_t endFrame = frames;
  if(frame != -1)
  {
    startFrame = frame;
    endFrame = frame+1;
  }

  newSliceSize = vox[0] * vox[1] * newChan * bpc;
  if (verbose) vvToolshed::initProgress(vox[2] * (endFrame-startFrame));
  raw.first();
  for (size_t f=startFrame; f<endFrame; ++f)
  {
    rd = raw.getData();
    newRaw = new uint8_t[newSliceSize * vox[2]];
    src = rd;
    dst = newRaw;
    for (ssize_t z=0; z<vox[2]; ++z)
    {
      for (ssize_t y=0; y<vox[1]; ++y)
      {
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          // Perform actual conversion:
          for (size_t i=0; i<newChan; ++i)
          {
            if (i < chan) memcpy(dst+i*bpc, src+i*bpc, bpc);
            else memset(dst+i*bpc, 0, bpc);
          }
          src += getBPV();
          dst += bpc * newChan;
        }
      }
      if (verbose) vvToolshed::printProgress(z + vox[2] * (f-startFrame));
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    raw.next();
  }
  // Adjust channel names:
  if (newChan != chan)
  {
    channelNames.resize(newChan);
  }
  chan = newChan;
}

//----------------------------------------------------------------------------
/** Delete a channel.
  @param channel channel index to delete [0 is first channel, valid: 1..numChannels-1]
  @param verbose true = print progress info
*/
void vvVolDesc::deleteChannel(size_t channel, bool verbose)
{
  uint8_t* newRaw;
  uint8_t* rd;
  uint8_t* src;
  uint8_t* dst;
  size_t newSliceSize;

  vvDebugMsg::msg(2, "vvVolDesc::deleteChannel()");

  if (channel >= chan) return;                    // this was easy!

  newSliceSize = vox[0] * vox[1] * (chan-1) * bpc;
  if (verbose) vvToolshed::initProgress(vox[2] * frames);
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    newRaw = new uint8_t[newSliceSize * vox[2]];
    src = rd;
    dst = newRaw;
    for (ssize_t z=0; z<vox[2]; ++z)
    {
      for (ssize_t y=0; y<vox[1]; ++y)
      {
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          // Perform actual conversion:
          for (size_t c=0; c<channel; ++c)
          {
            memcpy(dst+c*bpc, src+c*bpc, bpc);    // copy channels up to the one to delete
          }
          for (size_t c=(channel+1); c<chan; ++c)
          {
            memcpy(dst+(c-1)*bpc, src+c*bpc, bpc);// copy channels above the one to delete
          }
          src += getBPV();
          dst += bpc * (chan-1);
        }
      }
      if (verbose) vvToolshed::printProgress(z + vox[2] * f);
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    raw.next();
  }
  channelNames.erase(channelNames.begin() + channel);
  --chan;
}

//----------------------------------------------------------------------------
/** Bit-shifts all bytes of each voxel.
  @param bits number of bits to shift (>0 = right, <0 = left)
  @param verbose true = print progress info
*/
void vvVolDesc::bitShiftData(int bits, int frame, bool verbose)
{
  int  shift;
  uint8_t* rd;
  size_t sliceSize;
  size_t offset;

  vvDebugMsg::msg(2, "vvVolDesc::bitShiftData()");
  assert(bpc<=sizeof(unsigned long));                 // shift only works up to sizeof(long) byte per pixel
  if (bits==0) return;                            // done!

  sliceSize = getSliceBytes();
  shift = ts_max(bits, -bits);                    // find absolute value
  size_t startFrame=0;
  size_t endFrame=frames;
  if(frame != -1)
  {
    startFrame = frame;
    endFrame = frame+1;
  }
  if (verbose) vvToolshed::initProgress(vox[2] * (endFrame-startFrame));
  raw.first();
  for (size_t f=startFrame; f<endFrame; ++f)
  {
    rd = raw.getData();
    raw.next();
    for (ssize_t z=0; z<vox[2]; ++z)
    {
      for (ssize_t y=0; y<vox[1]; ++y)
        for (ssize_t x=0; x<vox[0]; ++x)
      {
        offset = x * getBPV() + y * vox[0] * getBPV() + z * sliceSize;
        unsigned long val = 0;
        for (size_t b=0; b<bpc*chan; b+=bpc)
        {
          switch(bpc) {
          case 1:
            val = *(uint8_t *)(rd+offset+b);
            break;
          case 2:
            val = *(uint16_t *)(rd+offset+b);
            break;
          case 4:
            val = *(uint32_t *)(rd+offset+b);
            break;
          }

          if (bits>0)
            val = val >> shift;
          else
            val = val << shift;

          switch(bpc) {
          case 1:
            *(uint8_t *)(rd+offset+b) = uint8_t(val);
            break;
          case 2:
            *(uint16_t *)(rd+offset+b) = uint16_t(val);
            break;
          case 4:
            *(uint32_t *)(rd+offset+b) = uint32_t(val);
            break;
          }
        }
      }
      if (verbose) vvToolshed::printProgress(z + vox[2] * (f-startFrame));
    }
  }
}

//----------------------------------------------------------------------------
/** Invert all bits of each scalar voxel value.
 */
void vvVolDesc::invert()
{
  uint8_t* ptr;                                     // pointer to currently worked on byte
  uint8_t* rd;

  vvDebugMsg::msg(2, "vvVolDesc::invert()");

  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    raw.next();
    ptr = &rd[0];
    for (ssize_t z=0; z<vox[2]; ++z)
      for (ssize_t y=0; y<vox[1]; ++y)
        for (ssize_t x=0; x<vox[0]; ++x)
          for (size_t b=0; b<bpc*chan; ++b)
          {
            *ptr = (uint8_t)(~(*ptr));
            ++ptr;
          }
  }
}

//----------------------------------------------------------------------------
/// Convert 24 bit RGB to 8 bit RGB332.
void vvVolDesc::convertRGB24toRGB8()
{
  uint8_t* newRaw;
  int pixel;
  int color;
  int shift[3] =                                  // 3 bit red, 3 bit green, 2 bit blue
  {
    3, 3, 2
  };
  uint8_t* rd;
  size_t newSliceSize;
  size_t oldSliceSize;

  vvDebugMsg::msg(2, "vvVolDesc::convertRGB24toRGB8()");
  assert(bpc==1 && chan==3);                      // cannot work on non-24bit-modes

  oldSliceSize = getSliceBytes();
  newSliceSize = vox[0] * vox[1];
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    newRaw = new uint8_t[vox[0] * vox[1] * vox[2]];
    for (ssize_t z=0; z<vox[2]; ++z)
      for (ssize_t y=0; y<vox[1]; ++y)
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          pixel = 0;
          for (int i=0; i<3; ++i)
          {
            color = rd[x * 3 + y * vox[0] * 3 + z * oldSliceSize + i];
            pixel &= 0xFF00;
            pixel += color;
            pixel <<= shift[i];
          }
          newRaw[x + y * vox[0] + z * newSliceSize] = (uint8_t)(pixel >> 8);
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    raw.next();
  }
  chan = 1;
}

//----------------------------------------------------------------------------
/** Flip voxel data along a coordinate axis.<PRE>
  Coordinate system:
      y
      |_x
    z/
  </PRE>
  Strategy:
  <UL>
  <LI>X Axis: Copy each line of voxels to a buffer and reversely copy voxels
      back to the data set one-by-one.</LI>
  <LI>Y Axis: Copy each line of voxels to a buffer and exchange the
line with the destination line.</LI>
<LI>Z Axis: Copy each slice of voxels to a buffer and exchange it
with the destination slice.</LI>
</UL>
@param axis axis along which to flip the volume data
*/
void vvVolDesc::flip(vvVecmath::AxisType axis)
{
  uint8_t* rd;
  uint8_t* voxelData;                             // temporary buffer for voxel data
  uint8_t* dst;                                   // destination pointer
  uint8_t* src;                                   // source pointer
  size_t lineSize;
  size_t sliceSize;

  vvDebugMsg::msg(2, "vvVolDesc::flip()");

  lineSize = vox[0] * getBPV();
  sliceSize = getSliceBytes();
  if (axis==vvVecmath::Z_AXIS) voxelData = new uchar[sliceSize];
  else voxelData = new uint8_t[lineSize];
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    switch (axis)
    {
      case vvVecmath::X_AXIS:
        dst = rd;
        for (ssize_t z=0; z<vox[2]; ++z)
          for (ssize_t y=0; y<vox[1]; ++y)
        {
          memcpy((void*)voxelData, (void*)dst, lineSize);
          src = voxelData + (vox[0]-1) * getBPV();
          for (ssize_t x=0; x<vox[0]; ++x)
          {
            memcpy(dst, src, getBPV());
            dst += getBPV();
            src -= getBPV();
          }
        }
        break;
      case vvVecmath::Y_AXIS:
        for (ssize_t z=0; z<vox[2]; ++z)
          for (ssize_t y=0; y<vox[1]/2; ++y)
        {
          src = rd + y * lineSize + z * sliceSize;
          dst = rd + (vox[1] - y - 1) * lineSize + z * sliceSize;
          memcpy((void*)voxelData, (void*)dst, lineSize);
          memcpy((void*)dst, (void*)src, lineSize);
          memcpy((void*)src, (void*)voxelData, lineSize);
        }
        break;
      case vvVecmath::Z_AXIS:
        for (ssize_t z=0; z<vox[2]/2; ++z)
        {
          dst = rd + z * sliceSize;
          src = rd + (vox[2]-z-1) * sliceSize;
          memcpy((void*)voxelData, (void*)dst, sliceSize);
          memcpy((void*)dst, (void*)src, sliceSize);
          memcpy((void*)src, (void*)voxelData, sliceSize);
        }
        break;
      default: break;
    }
    raw.next();
  }
  delete[] voxelData;
}

//----------------------------------------------------------------------------
/** Rotate voxel data about a coordinate axis.<PRE>
  Coordinate system:
      y
      |_x
    z/
  </PRE>
  @param axis axis about which to rotate the volume data
  @param dir  direction into which to rotate when looking at the origin
              from the positive half of the chosen coordinate axis (-1=left, 1=right)
*/
void vvVolDesc::rotate(vvVecmath::AxisType axis, int dir)
{
  uint8_t* rd;
  uint8_t* dst;                                   // destination pointer
  uint8_t* src;                                   // source pointer
  uint8_t* newRaw;                                // new volume data
  size_t newWidth, newHeight, newSlices;          // dimensions of rotated volume
  size_t x, y, z;
  size_t xpos, ypos, zpos;

  vvDebugMsg::msg(2, "vvVolDesc::rotate()");
  if (dir!=-1 && dir!=1) return;                  // validate direction

  // Compute the new volume size:
  switch (axis)
  {
    case vvVecmath::X_AXIS:
      newWidth  = vox[0];
      newHeight = vox[2];
      newSlices = vox[1];
      break;
    case vvVecmath::Y_AXIS:
      newWidth  = vox[2];
      newHeight = vox[1];
      newSlices = vox[0];
      break;
    case vvVecmath::Z_AXIS:
      newWidth  = vox[1];
      newHeight = vox[0];
      newSlices = vox[2];
      break;
    default:                                      // no change
      newWidth  = vox[0];
      newHeight = vox[1];
      newSlices = vox[2];
      break;
  }

  size_t frameSize = getFrameBytes();
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    newRaw = new uint8_t[frameSize];
    src = rd;
    switch (axis)
    {
      case vvVecmath::X_AXIS:
        for (y=0; y<newHeight; ++y)
        {
          if (dir>0) ypos = y;
          else       ypos = newHeight - 1 - y;
          for (z=0; z<newSlices; ++z)
          {
            if (dir>0) zpos = newSlices - 1 - z;
            else       zpos = z;
            for (x=0; x<newWidth; ++x)
            {
              dst = newRaw + getBPV() * (x + ypos * newWidth + zpos * newWidth * newHeight);
              memcpy((void*)dst, (void*)src, getBPV());
              src += getBPV();
            }
          }
        }
        break;
      case vvVecmath::Y_AXIS:
        for (x=0; x<newWidth; ++x)
        {
          if (dir>0) xpos = x;
          else       xpos = newWidth - 1 - x;
          for (y=0; y<newHeight; ++y)
            for (z=0; z<newSlices; ++z)
          {
            if (dir>0) zpos = newSlices - 1 - z;
            else       zpos = z;
            dst = newRaw + getBPV() * (xpos + y * newWidth + zpos * newWidth * newHeight);
            memcpy((void*)dst, (void*)src, getBPV());
            src += getBPV();
          }
        }
        break;
      case vvVecmath::Z_AXIS:
        for (z=0; z<newSlices; ++z)
          for (x=0; x<newWidth; ++x)
        {
          if (dir>0) xpos = newWidth - 1 - x;
          else       xpos = x;
          for (y=0; y<newHeight; ++y)
          {
            if (dir>0) ypos = y;
            else       ypos = newHeight - 1 - y;
            dst = newRaw + getBPV() * (xpos + ypos * newWidth + z * newWidth * newHeight);
            memcpy((void*)dst, (void*)src, getBPV());
            src += getBPV();
          }
        }
        break;
      default: break;
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
    raw.next();
  }
  vox[0] = newWidth;
  vox[1] = newHeight;
  vox[2] = newSlices;
}

//----------------------------------------------------------------------------
/** Convert 24 bit RGB planar (RRRR..., GGGG..., BBBB...)
   to 24 bit RGB interleaved (RGB, RGB, RGB, ...).
   Planar format must be repeated in each volume frame.
*/
void vvVolDesc::convertRGBPlanarToRGBInterleaved(int frame)
{
  uint8_t* raw;
  uint8_t* tmpData;

  vvDebugMsg::msg(2, "vvVolDesc::convertRGBPlanarToRGBInterleaved()");
  assert(bpc==1 && chan==3);                      // this routine works only on RGB volumes

  size_t frameSize = getFrameBytes();
  size_t voxels = getFrameVoxels();
  tmpData = new uint8_t[frameSize];
  size_t startFrame = 0;
  size_t endFrame = frames;
  if(frame != -1)
  {
    startFrame = frame;
    endFrame = frame+1;
  }
  for (size_t f=startFrame; f<endFrame; ++f)
  {
    raw = getRaw(f);
    for (size_t i=0; i<voxels; ++i)
      for (size_t c=0; c<3; ++c)
        tmpData[i * 3 + c] = raw[c * voxels + i];
    memcpy(raw, tmpData, frameSize);
  }
  delete[] tmpData;
}

//----------------------------------------------------------------------------
/** Reverse endianness of bytes in voxels.
  <UL>
    <LI>bpc=1: nothing to be done</LI>
    <LI>bpc=2: high and low byte are swapped</LI>
    <LI>bpc=4: byte order is reversed</LI>
  </UL>
*/
void vvVolDesc::toggleEndianness(int frame)
{
  uint8_t* rd;
  size_t    rowOffset, sliceOffset, voxelOffset, channelOffset;
  uint8_t  buffer;

  vvDebugMsg::msg(2, "vvVolDesc::toggleEndianness()");
  if (bpc==1) return;                             // done

  size_t sliceSize = getSliceBytes();
  raw.first();
  size_t startFrame=0;
  size_t endFrame=frames;

  if(frame != -1)
  {
    startFrame = frame;
    endFrame = frame+1;
  }
  const int bpv = getBPV();
  const size_t n = vox[0] * vox[1] * vox[2] * bpv / bpc;
  for (size_t f=startFrame; f<endFrame; ++f)
  {
    rd = getRaw(f);

    if (bpc == 2) {
      uint16_t *r = reinterpret_cast<uint16_t *>(rd);
      for (size_t i=0; i<n; ++i) {
        r[i] = byte_swap<little_endian, big_endian, uint16_t>(r[i]);
      }
    } else if (bpc == 4) {
      uint32_t *r = reinterpret_cast<uint32_t *>(rd);
      for (size_t i=0; i<n; ++i) {
        r[i] = byte_swap<little_endian, big_endian, uint32_t>(r[i]);
      }
    }

#if 0
    for (ssize_t z=0; z<vox[2]; ++z)
    {
      sliceOffset = z * sliceSize;
      for (ssize_t y=0; y<vox[1]; ++y)
      {
        rowOffset = sliceOffset + y * vox[0] * bpv;
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          voxelOffset = x * bpv + rowOffset;

          for (size_t c=0; c<chan; ++c)
          {
            channelOffset = voxelOffset + c * bpc;

            // Swap first and last byte of voxel:
            buffer = rd[channelOffset];
            rd[channelOffset] = rd[channelOffset + bpc - 1];
            rd[channelOffset + bpc - 1] = buffer;

            // For float voxels also swap middle bytes:
            if (bpc==4)
            {
              buffer = rd[channelOffset + 1];
              rd[channelOffset + 1] = rd[channelOffset + 2];
              rd[channelOffset + 2] = buffer;
            }
          }
        }
      }
    }
#endif
  }
}

//----------------------------------------------------------------------------
/** Toggle sign of data values.
  <UL>
    <LI>bpv=1: invert most significant bit</LI>
    <LI>bpv=2: invert most significant bit</LI>
    <LI>bpv=4: negate floating point value</LI>
  </UL>
*/
void vvVolDesc::toggleSign(int /* frame */)
{
  uint8_t* rd;
  float  val;

  vvDebugMsg::msg(2, "vvVolDesc::toggleSign()");

  size_t frameVoxels = getFrameVoxels();
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    for (size_t i=0; i<frameVoxels*chan; ++i)
    {
      switch(bpc)
      {
        case 1:
        case 2:
          *rd ^= 0x80;                            // bitwise exclusive or with 10000000b to negate MSB
          break;
        case 4:
          val = *((float*)rd);
          val = -val;
          *((float*)rd) = val;
          break;
      }
      rd += bpc;
    }
    raw.next();
  }
}

//----------------------------------------------------------------------------
/** Returns true if a specific channel is nonzero in any voxel of the volume.
  @param m channel index (0=first)
*/
bool vvVolDesc::isChannelUsed(size_t m)
{
  uint8_t* rd;

  vvDebugMsg::msg(2, "vvVolDesc::isChannelUsed()");
  assert(m<chan);

  size_t frameSize = getFrameVoxels();
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    for (size_t i=0; i<frameSize; ++i)
    {
      switch(bpc)
      {
        case 1: if (rd[i + m] != 0) return true;
        case 2: if (rd[i + m] != 0 || rd[i + m + 1] != 0) return true;
        case 4: if (*((float*)rd) != 0.0f) return true;
      }
    }
    raw.next();
  }
  return false;
}

//----------------------------------------------------------------------------
/** Crop each volume of the animation to a sub-volume.
  @param x,y,z  coordinates of top-left-front corner of sub-volume
  @param w,h,s  width, height, and number of slices of sub-volume
*/
void vvVolDesc::crop(ssize_t x, ssize_t y, ssize_t z, ssize_t w, ssize_t h, ssize_t s)
{
  ssize_t j, i;
  uint8_t* newRaw;
  uint8_t* rd;
  ssize_t xmin, xmax, ymin, ymax, zmin, zmax;
  ssize_t newWidth, newHeight, newSlices;
  size_t newSliceSize;
  size_t oldSliceSize;
  uint8_t *src, *dst;

  vvDebugMsg::msg(2, "vvVolDesc::crop()");

  // Find minimum and maximum values for crop:
  xmin = ts_max(ssize_t(0), ts_min(x, vox[0]-1, x + w - 1));
  ymin = ts_max(ssize_t(0), ts_min(y, vox[1]-1, y + h - 1));
  zmin = ts_max(ssize_t(0), ts_min(z, vox[2]-1, z + s - 1));
  xmax = ts_min(vox[0]-1, ts_max(x, x + w - 1));
  ymax = ts_min(vox[1]-1, ts_max(y, y + h - 1));
  zmax = ts_min(vox[2]-1, ts_max(z, z + s - 1));

  // Set new volume dimensions:
  newWidth  = xmax - xmin + 1;
  newHeight = ymax - ymin + 1;
  newSlices = zmax - zmin + 1;

  // Now cropping can be done:
  oldSliceSize = getSliceBytes();
  newSliceSize = newWidth * newHeight * getBPV();
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    newRaw = new uint8_t[newSliceSize * newSlices];
    for (j=0; j<newSlices; ++j)
      for (i=0; i<newHeight; ++i)
    {
      src = rd + (j + zmin) * oldSliceSize +
        (i + ymin) * vox[0] * getBPV() + xmin * getBPV();
      dst = newRaw + j * newSliceSize + i * newWidth * getBPV();
      memcpy(dst, src, newWidth * getBPV());
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
    raw.next();
  }

  // Set new sizes:
  vox[0] = newWidth;
  vox[1] = newHeight;
  vox[2] = newSlices;
}

//----------------------------------------------------------------------------
/** Remove time steps at start and end of volume animation.
  @param start first time step to keep in the sequence [0..timesteps-1]
  @param steps number of steps to keep
*/
void vvVolDesc::cropTimesteps(size_t start, size_t steps)
{
  raw.first();

  // Remove steps before the desired range:
  for (size_t i=0; i<start; ++i)
    raw.remove();

  // Remove steps after the desired range:
  raw.last();
  for (size_t i=0; i<frames-start-steps; ++i)
    raw.remove();

  frames = raw.count();
}

//----------------------------------------------------------------------------
/** Resize each volume of the animation. The real voxel size parameters
  are adjusted accordingly.
  @param w,h,s   new width, height, and number of slices
  @param ipt     interpolation type to use for resampling
  @param verbose true = verbose mode
*/
void vvVolDesc::resize(ssize_t w, ssize_t h, ssize_t s, InterpolationType ipt, bool verbose)
{
  uint8_t* newRaw;                                  // pointer to new volume data
  uint8_t* rd;
  size_t newSliceSize, newFrameSize;
  size_t oldSliceVoxels;
  uint8_t *src, *dst;
  ssize_t ix, iy, iz;                             // integer source voxel coordinates
  float fx, fy, fz;                               // floating point source voxel coordinates
  uint8_t interpolated[4];                        // interpolated voxel values

  vvDebugMsg::msg(2, "vvVolDesc::resize()");

  // Validate resize parameters:
  if (w<=0 || h<=0 || s<=0) return;
  if (w==vox[0] && h==vox[1] && s==vox[2]) return;// already done

  // Now resizing can be done:
  oldSliceVoxels = getSliceVoxels();
  newSliceSize = w * h * getBPV();
  newFrameSize = newSliceSize * s;
  if (verbose) vvToolshed::initProgress(s * frames);
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    newRaw = new uint8_t[newFrameSize];
    dst = newRaw;

    // Traverse destination data:
    for (ssize_t z=0; z<s; ++z)
    {
      for (ssize_t y=0; y<h; ++y)
        for (ssize_t x=0; x<w; ++x)
      {
        // Compute source coordinates of current destination voxel:
        if (ipt==TRILINEAR)                       // trilinear interpolation
        {
          // Compute source coordinates of current destination voxel:
          fx = (float)x / (float)(w-1) * (float)(vox[0]-1);
          fy = (float)y / (float)(h-1) * (float)(vox[1]-1);
          fz = (float)z / (float)(s-1) * (float)(vox[2]-1);
          trilinearInterpolation(f, fx, fy, fz, interpolated);

          // Copy interpolated voxel data to destination voxel:
          memcpy(dst, interpolated, getBPV());
        }
        else                                      // nearest neighbor interpolation
        {
          // Compute source coordinates of current destination voxel:
          if (w>1) ix = x * (vox[0]-1)  / (w-1);
          else     ix = 0;
          if (h>1) iy = y * (vox[1]-1) / (h-1);
          else     iy = 0;
          if (s>1) iz = z * (vox[2]-1) / (s-1);
          else     iz = 0;
          ix = ts_clamp(ix, ssize_t(0), vox[0]-1);
          iy = ts_clamp(iy, ssize_t(0), vox[1]-1);
          iz = ts_clamp(iz, ssize_t(0), vox[2]-1);

          // Copy source voxel data to destination voxel:
          src = rd + getBPV() * (ix + iy * vox[0] + iz * oldSliceVoxels);
          memcpy(dst, src, getBPV());
        }
        dst += getBPV();
      }
      if (verbose) vvToolshed::printProgress(z + s * f);
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
    raw.next();
  }

  // Adjust voxel size:
  dist[0] *= float(vox[0]) / float(w);
  dist[1] *= float(vox[1]) / float(h);
  dist[2] *= float(vox[2]) / float(s);

  // Use new size:
  vox[0] = w;
  vox[1] = h;
  vox[2] = s;
}

//----------------------------------------------------------------------------
/** Shift each volume of the animation by a number of voxels.
  Rotary boundary conditions are applied.
  Strategy: The volume is shifted slicewise so that only one
  volume slice has to be duplicated in memory.
  @param sx,sy,sz  shift amounts. Positive values shift into the
                   positive axis direction.
*/
void vvVolDesc::shift(int sx, int sy, int sz)
{
  uint8_t* rd;
  uint8_t* newRaw;                                // shifted volume data
  uint8_t* src;
  uint8_t* dst;
  size_t lineSize, sliceSize, frameSize;
  int sval[3];                                    // shift amount

  vvDebugMsg::msg(2, "vvVolDesc::shift()");

  // Consider rotary boundary conditions and make shift values positive:
  if (sx==0 && sy==0 && sz==0) return;

  sval[0] =  sx % vox[0];
  sval[1] = -sy % vox[1];
  sval[2] = -sz % vox[2];
  if (sval[0]<0) sval[0] += vox[0];
  if (sval[1]<0) sval[1] += vox[1];
  if (sval[2]<0) sval[2] += vox[2];

  // Now shifting starts:
  lineSize  = vox[0] * getBPV();
  sliceSize = getSliceBytes();
  frameSize = getFrameBytes();
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    for (size_t i=0; i<3; ++i)
    {
      raw.makeCurrent(f);
      rd = raw.getData();

      if (sval[i] > 0)
      {
        newRaw = new uint8_t[frameSize];
        switch (i)
        {
          case 0:                                 // x shift
            src = rd;
            for (ssize_t z=0; z<vox[2]; ++z)
              for (ssize_t y=0; y<vox[1]; ++y)
                for (ssize_t x=0; x<vox[0]; ++x)
                {
                  dst = newRaw + z * sliceSize + y * lineSize + ((x + sval[0]) % vox[0]) * getBPV();
                  memcpy(dst, src, getBPV());
                  src += getBPV();
                }
          break;
          case 1:                                 // y shift
            src = rd;
            for (ssize_t z=0; z<vox[2]; ++z)
              for (ssize_t y=0; y<vox[1]; ++y)
            {
              dst = newRaw + z * sliceSize + ((y + sval[1]) % vox[1]) * lineSize;
              memcpy(dst, src, lineSize);
              src += lineSize;
            }
            break;
          default:
          case 2:                                 // z shift
            src = rd;
            for (ssize_t z=0; z<vox[2]; ++z)
            {
              dst = newRaw + ((z + sval[2]) % vox[2]) * sliceSize;
              memcpy(dst, src, sliceSize);
              src += sliceSize;
            }
            break;
        }
        raw.remove();
        if (f==0) raw.insertBefore(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
        else raw.insertAfter(newRaw, vvSLNode<uchar*>::ARRAY_DELETE);
      }
    }
    raw.next();
  }
}

//----------------------------------------------------------------------------
/** Inverts innermost and outermost voxel loops.
*/
void vvVolDesc::convertVoxelOrder()
{
  uint8_t* raw;
  uint8_t* tmpData;
  uint8_t* src;
  uint8_t* dst;

  vvDebugMsg::msg(2, "vvVolDesc::convertVoxelOrder()");

  size_t frameSize = getFrameBytes();
  tmpData = dst = new uint8_t[frameSize];
  for (size_t f=0; f<frames; ++f)
  {
    raw = src = getRaw(f);
    for (ssize_t x=0; x<vox[0]; ++x)
      for (ssize_t y=0; y<vox[1]; ++y)
        for (ssize_t z=0; z<vox[2]; ++z)
        {
          dst = tmpData + getBPV() * (x + (vox[1] - y - 1) * vox[0] + z * vox[0] * vox[1]);
          memcpy(dst, src, getBPV());
          src += getBPV();                        // skip to next voxel
        }
    memcpy(raw, tmpData, frameSize);
  }
  delete[] tmpData;
}

//----------------------------------------------------------------------------
/** Convert COVISE volume data to Virvo format.
 The major difference is that in COVISE the innermost and the outermost
 voxel loops are inverted.
*/
void vvVolDesc::convertCoviseToVirvo()
{
  uint8_t* raw;
  uint8_t* tmpData;
  size_t srcIndex;                                   // index into COVISE volume array
  uint8_t* ptr;

  vvDebugMsg::msg(2, "vvVolDesc::convertCoviseToVirvo()");

  size_t frameSize = getFrameBytes();
  tmpData = new uint8_t[frameSize];
  for (size_t f=0; f<frames; ++f)
  {
    raw = getRaw(f);
    ptr = tmpData;
    for (ssize_t z=0; z<vox[2]; ++z)
      for (ssize_t y=0; y<vox[1]; ++y)
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          srcIndex = getBPV() * ((vox[2]-z-1)+ (vox[1]-y-1) * vox[2] + x * vox[1] * vox[2]);
          memcpy(ptr, raw + srcIndex, getBPV());
          ptr += getBPV();                        // skip to next voxel
        }
    memcpy(raw, tmpData, frameSize);
  }
  delete[] tmpData;

#if defined(__linux__) || defined(LINUX)
  if (bpc==1 && chan==4) toggleEndianness();      // RGBA data are transferred as packed colors, which are integers
#endif
}

//----------------------------------------------------------------------------
/** Convert Virvo volume data to COVISE format.
 The major difference is that the innermost and the outermost
 voxel loops are inverted.
*/
void vvVolDesc::convertVirvoToCovise()
{
  uint8_t* raw;
  uint8_t* tmpData;
  uint8_t* ptr;
  size_t    dstIndex;                                // index into COVISE volume array

  vvDebugMsg::msg(2, "vvVolDesc::convertVirvoToCovise()");

  size_t frameSize = getFrameBytes();
  tmpData = new uint8_t[frameSize];
  for (size_t f=0; f<frames; ++f)
  {
    raw = getRaw(f);
    ptr = raw;
    for (ssize_t z=0; z<vox[2]; ++z)
      for (ssize_t y=0; y<vox[1]; ++y)
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          dstIndex = getBPV() * (x * vox[1] * vox[2] + (vox[1]-y-1) * vox[2] + (vox[2]-z-1));
          memcpy(tmpData + dstIndex, ptr, getBPV());
          ptr += getBPV();                        // skip to next voxel
        }
    memcpy(raw, tmpData, frameSize);
  }
  delete[] tmpData;
}

//----------------------------------------------------------------------------
/** Convert Virvo volume data to OpenGL format.
 The OpenGL format is as follows: counting starts at the backmost slice bottom left,
 it continues to the right, then up and then to the front.
*/
void vvVolDesc::convertVirvoToOpenGL()
{
  uint8_t* raw;
  uint8_t* tmpData;
  uint8_t* ptr;
  size_t    dstIndex;                                // index into OpenGL volume array

  vvDebugMsg::msg(2, "vvVolDesc::convertVirvoToOpenGL()");

  size_t frameSize = getFrameBytes();
  tmpData = new uint8_t[frameSize];
  for (size_t f=0; f<frames; ++f)
  {
    raw = getRaw(f);
    ptr = raw;
    for (ssize_t z=0; z<vox[2]; ++z)
      for (ssize_t y=0; y<vox[1]; ++y)
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          dstIndex = getBPV() * (x + (vox[1] - y - 1) * vox[0] + (vox[2] - z - 1) * vox[0] * vox[1]);
          memcpy(tmpData + dstIndex, ptr, getBPV());
          ptr += getBPV();
        }
    memcpy(raw, tmpData, frameSize);
  }
  delete[] tmpData;
}

//----------------------------------------------------------------------------
/** Convert OpenGL volume data to Virvo format.
 The OpenGL format is as follows: counting starts at the backmost slice bottom left,
 it continues to the right, then up and then to the front.
*/
void vvVolDesc::convertOpenGLToVirvo()
{
  uint8_t* raw;
  uint8_t* tmpData;
  uint8_t* ptr;
  size_t    dstIndex;                                // index into Virvo volume array

  vvDebugMsg::msg(2, "vvVolDesc::convertOpenGLToVirvo()");

  size_t frameSize = getFrameBytes();
  tmpData = new uint8_t[frameSize];
  for (size_t f=0; f<frames; ++f)
  {
    raw = getRaw(f);
    ptr = raw;
    for (ssize_t z=0; z<vox[2]; ++z)
      for (ssize_t y=0; y<vox[1]; ++y)
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          dstIndex = getBPV() * (x + (vox[1] - y - 1) * vox[0] + (vox[2] - z - 1)  * vox[0] * vox[1]);
          memcpy(tmpData + dstIndex, ptr, getBPV());
          ptr += getBPV();
        }
    memcpy(raw, tmpData, frameSize);
  }
  delete[] tmpData;
}

//----------------------------------------------------------------------------
/** Creates an icon from passed RGBA data.
 The caller must delete the passed rgba array.
 @param s    icon size in pixels (size = width = height), if 0 icon will be deleted
 @param rgb  icon image data array (size * size * 4 bytes expected)
*/
void vvVolDesc::makeIcon(size_t s, const uint8_t* rgba)
{
  const size_t BPP = 4;
  vvDebugMsg::msg(2, "vvVolDesc::makeIcon(2)");

  if (s<=0)                                       // delete icon
  {
    iconSize = 0;
    if (iconData!=NULL) delete[] iconData;
    iconData = NULL;
  }
  else
  {
    iconSize = s;
    size_t iconBytes = iconSize * iconSize * BPP;
    iconData = new uint8_t[iconBytes];
    memcpy(iconData, rgba, iconBytes);
  }
}

//----------------------------------------------------------------------------
/** Make an icon by projecting all slices on top of each other (back to
  front), using MIP.
  @param size icon edge length [pixels]
*/
void vvVolDesc::makeIcon(size_t size)
{
  vvVolDesc* tmpVD;
  size_t iconBytes;

  vvDebugMsg::msg(2, "vvVolDesc::makeIcon(1)");

  // Create temporary volume if necessary:
  if (bpc==1) tmpVD = this;
  else tmpVD = new vvVolDesc(this, 0);
  tmpVD->convertBPC(1);

  // Prepare icon data:
  iconSize = size;
  iconBytes = iconSize * iconSize * ICON_BPP;
  delete[] iconData;
  iconData = new uint8_t[iconBytes];

  // Compute icon image:
  uint8_t* tmpSlice = new uint8_t[iconBytes];
  memset(iconData, 0, iconBytes);
  uint8_t* raw = tmpVD->getRaw();
  for (ptrdiff_t i=tmpVD->vox[2]-1; i>=0; --i)
  {
    // Resample current volume slice to temporary image of icon size:
    vvToolshed::resample(raw + tmpVD->getSliceBytes() * i,
      tmpVD->vox[0], tmpVD->vox[1], tmpVD->getBPV(), tmpSlice, iconSize, iconSize, ICON_BPP);

    // Add temporary image to icon:
    vvToolshed::blendMIP(tmpSlice, iconSize, iconSize, ICON_BPP, iconData);
  }
  delete[] tmpSlice;
  if (tmpVD != this) delete tmpVD;
}

//----------------------------------------------------------------------------
/** Convert flat data to a sphere.
  The sphere coordinate equations are taken from Bronstein page 154f.
  The routine traverses the destination voxels, reversely computes
  their locations on the planar data set, and copies these voxel
  values. The source volume is mapped on a sphere which can be
  imagined to be located 'behind' the source volume: Up remains
  up, the 'seam' is located at the front of the sphere. Lower
  z values become located a more remote regions from the sphere
  center.<P>
  The used coordinate system is:<PRE>
        z
/
/___x
|
|
y
</PRE>
R is the distance from the current voxel to the volume center.<BR>
Phi is the angle in the x/z axis, starting at (0|0|-1), going
past (1|0|0) and (0|0|1) and back to (0|0|-1).<BR>
Theta is the angle from the position vector of the current
voxel to the vector (0|-1|0).
@param outer   outer sphere diameter [voxels]
@param inner   inner sphere diameter [voxels]
@param ipt     interpolation type
@param verbose true = verbose mode
*/
void vvVolDesc::makeSphere(size_t outer, size_t inner, InterpolationType ipt, bool verbose)
{
  uint8_t* rd;                                    // raw data of current source frame
  vvVector3 center;                               // sphere center position [voxel space]
  vvVector3 v;                                    // currently processed voxel coordinates [voxel space]
  uint8_t* newRaw;                                // raw data of current destination frame
  float dist;                                     // distance from voxel to sphere center
  float phi;                                      // angle in x/z plane
  float theta;                                    // angle to y axis
  uint8_t *src, *dst;                             // source and destination volume data
  float radius;                                   // outer radius of sphere [voxels]
  float core;                                     // core radius [voxels]
  size_t newFrameSize;                            // sphere volume frame size [voxels]
  size_t sliceVoxels;                             // number of voxels per slice in source volume
  float sx, sy, sz;                               // coordinates in source volume
  float ringSize;                                 // precomputed ring size
  uint8_t interpolated[4];                        // interpolated voxel values

  vvDebugMsg::msg(2, "vvVolDesc::makeSphere()");

  newFrameSize = outer * outer * outer * getBPV();
  if (outer>1)
    radius = (float)(outer-1) / 2.0f;
  else
    radius = 0.5f;
  if (inner>1)
    core = (float)(inner-1) / 2.0f;
  else
    core = 0.0f;
  ringSize = radius - core;
  if (ringSize<1.0f) ringSize = 1.0f;             // prevent division by zero later on
  center.set(radius, radius, radius);
  sliceVoxels = vox[0] * vox[1];
  if (verbose) vvToolshed::initProgress(outer * frames);
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    newRaw = new uint8_t[newFrameSize];
    dst = newRaw;

    // Traverse destination data:
    for (size_t z=0; z<outer; ++z)
    {
      for (size_t y=0; y<outer; ++y)
        for (size_t x=0; x<outer; ++x)
      {
        // Compute sphere coordinates of current destination voxel:
        v.set((float)x, (float)y, (float)z);
        v.sub(center);
        v[1] = -v[1];                             // adapt to vvVecmath coordinate system
        v[2] = -v[2];                             // adapt to vvVecmath coordinate system
        v.getSpherical(&dist, &phi, &theta);

        // Map sphere coordinates to planar coordinates in source volume:
        if (dist<=radius && dist>=core)           // is voxel within source volume?
        {
          // Compute source coordinates of current destination voxel:
          sx = phi / (2.0f * VV_PI) * (float)vox[0];
          sy = theta / VV_PI * (float)vox[1];
          sz = (float)vox[2] - 1.0f - ((dist-core) / ringSize * (float)vox[2]);
          if (ipt==TRILINEAR)                     // trilinear interpolation
          {
            trilinearInterpolation(f, sx, sy, sz, interpolated);
            memcpy(dst, interpolated, getBPV());
          }
          else                                    // nearest neighbor
          {
            sx = ts_clamp(sx, 0.0f, (float)(vox[0]-1));
            sy = ts_clamp(sy, 0.0f, (float)(vox[1]-1));
            sz = ts_clamp(sz, 0.0f, (float)(vox[2]-1));
            src = rd + getBPV() * ((size_t)sx + (size_t)sy * vox[0] + (size_t)sz * sliceVoxels);
            memcpy(dst, src, getBPV());
          }
        }
        else memset(dst, 0, getBPV());            // outside of sphere

        dst += getBPV();
      }
      if (verbose) vvToolshed::printProgress(z + outer * f);
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    raw.next();
  }
  vox[0] = vox[1] = vox[2] = outer;
}

//----------------------------------------------------------------------------
/** Display one line of volume information data.
  @param desc   description text, NULL for none
*/
void vvVolDesc::printInfoLine(const char* desc)
{
  std::string str;

  if (desc!=NULL) std::cerr << desc << " ";

  makeInfoString(&str);
  std::cerr << str << std::endl;
}

//----------------------------------------------------------------------------
/** Expects an array of 256 _allocated_ char values.
 */
void vvVolDesc::makeInfoString(std::string* infoString)
{
  char* basename = new char[strlen(getFilename()) + 1];
  vvToolshed::extractFilename(basename, getFilename());

  std::stringstream buf;
  buf << basename << ": " << frames << " ";
  if (frames != 1)
  {
    buf << "frames";
  }
  else
  {
    buf << "frame";
  }
  buf << ", " << vox[0] << " x " << vox[1] << " x " << vox[2] << " ";
  if (frames > 1)
  {
    buf << "voxels per frame";
  }
  else
  {
    buf << "voxels";
  }
  buf << ", " << bpc << " ";
  if (bpc != 1)
  {
    buf << "bytes";
  }
  else
  {
    buf << "byte";
  }
  buf << " per channel" << ", " << chan << " ";
  if (chan != 1)
  {
    buf << "channels";
  }
  else
  {
    buf << "channel";
  }
  *infoString = buf.str();

  delete[] basename;
}

//----------------------------------------------------------------------------
/** Expects an array of 256 _allocated_ char values.
 */
void vvVolDesc::makeShortInfoString(char* infoString)
{
  sprintf(infoString, "%d %s, %d x %d x %d, %d x %d",
    static_cast<int32_t>(frames), (frames!=1) ? "frames" : "frame",
    static_cast<int32_t>(vox[0]), static_cast<int32_t>(vox[1]),
    static_cast<int32_t>(vox[2]), static_cast<int32_t>(bpc), static_cast<int32_t>(chan));
}

//----------------------------------------------------------------------------
/// Display verbose volume information from file header.
void vvVolDesc::printVolumeInfo()
{
  cerr << "File name:                         " << filename << endl;
  cerr << "Volume size [voxels]:              " << vox[0] << " x " << vox[1] << " x " << vox[2] << endl;
  cerr << "Number of frames:                  " << frames << endl;
  cerr << "Bytes per channel:                 " << bpc << endl;
  cerr << "Channels:                          " << chan << endl;
  cerr << "Voxels per frame:                  " << getFrameVoxels() << endl;
  cerr << "Data bytes per frame:              " << getFrameBytes() << endl;
  if (frames > 1)
  {
    cerr << "Voxels total:                      " << getMovieVoxels() << endl;
    cerr << "Data bytes total:                  " << getMovieBytes() << endl;
  }
  cerr << "Sample distances:                  " << setprecision(3) << dist[0] << " x " << dist[1] << " x " << dist[2] << endl;
  cerr << "Time step duration [s]:            " << setprecision(3) << dt << endl;
  cerr << "Physical data range:               " << real[0] << " to " << real[1] << endl;
  cerr << "Object location [mm]:              " << pos[0] << ", " << pos[1] << ", " << pos[2] << endl;
  cerr << "Icon stored:                       " << ((iconSize>0) ? "yes" : "no") << endl;
  if (iconSize>0)
  {
    cerr << "Icon size:                         " << iconSize << " x " << iconSize << " pixels" << endl;
  }
}

//----------------------------------------------------------------------------
/// Display statistics about the volume data.
void vvVolDesc::printStatistics()
{
  float scalarMin, scalarMax;
  float mean, variance, stdev;
  float zeroVoxels, numTransparent;

  for (size_t c=0; c<chan; ++c)
  {
    if (chan>1) cerr << "Channel " << c+1 << endl;
    findMinMax(0, scalarMin, scalarMax);
    calculateDistribution(0, c, mean, variance, stdev);
    cerr << "Scalar value range:                " << scalarMin << " to " << scalarMax << endl;
    if (bpc<3)  // doesn't work with floats
    {
      cerr << "  Number of different data values in channel " << c+1 << ": " << findNumUsed(c) << endl;
    }
    zeroVoxels = 100.0f * findNumValue(0,0) / getFrameVoxels();
    numTransparent = 100.0f * findNumTransparent(0) / getFrameVoxels();
    cerr << "  Zero voxels in first frame:        " << setprecision(4) << zeroVoxels << " %" << endl;
    cerr << "  Transparent voxels in first frame: " << setprecision(4) << numTransparent << " %" << endl;
    cerr << "  Mean in first frame:               " << setprecision(4) << mean << endl;
    cerr << "  Variance in first frame:           " << setprecision(4) << variance << endl;
    cerr << "  Standard deviation in first frame: " << setprecision(4) << stdev << endl;
  }
}

//----------------------------------------------------------------------------
/** Display histogram as text. Omit non-occurring values.
  @param frame   frame to compute histogram for (-1 for all frames)
  @param channel channel to compute histogram for (0=first)
*/
void vvVolDesc::printHistogram(int frame, size_t channel)
{
  int* hist;
  int i;

  int buckets[1] = {32};

  hist = new int[buckets[0]];
  makeHistogram(frame, channel, 1, buckets, hist, real[0], real[1]);
  for (i=0; i<buckets[0]; ++i)

  {
    if (hist[i] > 0) cerr << i << ": " << hist[i] << endl;
  }
  delete[] hist;
}

//----------------------------------------------------------------------------
/** Display the voxel data as hex numbers.
  @param frame number of frame in dataset
  @param slice number of slice to display
  @param width number of voxels to display per row (starting left, 0 for all voxels)
  @param height number of voxels to display per column (starting at top, 0 for all voxels))
*/
void vvVolDesc::printVoxelData(int frame, ssize_t slice, ssize_t width, ssize_t height)
{
  uint8_t* raw;                                   // pointer to raw volume data
  ssize_t nx,ny;                                  // number of voxels to display per row/column
  int val;

  raw = getRaw(frame);
  raw += slice * getSliceBytes();                 // move pointer to beginning of slice to display
  if (width<=0) nx = vox[0];
  else nx = ts_min(width, vox[0]);
  if (height<=0) ny = vox[1];
  else ny = ts_min(height, vox[1]);
  cerr << "Voxel data of frame " << frame << ":" << endl;
  for (ssize_t y=0; y<ny; ++y)
  {
    cerr << "Row " << y << ": ";
    for (ssize_t x=0; x<nx; ++x)
    {
      cerr << "(";
      for (size_t m=0; m<chan; ++m)
      {
        switch (bpc)
        {
          default:
          case 1:
            val = int(raw[getBPV() * (x + y * vox[0]) + m]);
            cerr << val << " ";
            break;
          case 2:
            val = 256 * int(raw[getBPV() * (x + y * vox[0]) + m*2]) +
              int(raw[getBPV() * (x + y * vox[0]) + m*2 + 1]);
            cerr << val << " ";
            break;
          case 4:
            cerr << "{" << int(raw[getBPV() * (x + y * vox[0]) + m*4]) << "," <<
              int(raw[getBPV() * (x + y * vox[0]) + m*4 + 1]) << "," <<
              int(raw[getBPV() * (x + y * vox[0]) + m*4 + 2]) << "," <<
              int(raw[getBPV() * (x + y * vox[0]) + m*4 + 3]) << "} ";
            break;
        }
      }
      cerr << ") ";
    }
    cerr << endl;
  }
}

//----------------------------------------------------------------------------
/** Perform a trilinear interpolation.
  Neighboring vertex indices:<PRE>
      4____ 7
     /___ /|
   0|   3| |
    | 5  | /6
    |/___|/
    1    2
  </PRE>
  @param f      volume animation frame
  @param x,y,z  coordinates for which the interpolation must be performed
@param result interpolated voxel data. This must be a pointer to
an _allocated_ memory space of bpc * chan bytes.
*/
void vvVolDesc::trilinearInterpolation(size_t f, float x, float y, float z, uint8_t* result)
{
  uint8_t* rd;
  uint8_t* neighbor[8];                           // pointers to neighboring voxels
  float val[8];                                   // neighboring voxel values
  vvssize3 tfl;                                   // coordinates of neighbor 0 (top-front-left)
  float dist[3];                                  // distance to neighbor 0
  size_t lineSize, sliceSize;
  int interpolated;                               // interpolated value

  vvDebugMsg::msg(3, "vvVolDesc::trilinearInterpolation()");

  if (vox[0]<2 || vox[1]<2 || vox[2]<2) return;   // done!

  // Check for valid frame index:
  if (f>=frames) return;

  // Constrain voxel position to the valid region:
  x = ts_clamp(x, 0.0f, (float)(vox[0]-1));
  y = ts_clamp(y, 0.0f, (float)(vox[1]-1));
  z = ts_clamp(z, 0.0f, (float)(vox[2]-1));

  // Compute coordinates of neighbor 0:
  tfl[0] = (size_t)x;
  tfl[1] = (size_t)y;
  tfl[2] = (size_t)z;

  // Compute distance to neighbor 0:
  if (tfl[0] < (vox[0]-1))
    dist[0] = x - (float)tfl[0];
  else                                            // border values need special treatment
  {
    --tfl[0];
    dist[0] = 1.0f;
  }
  if (tfl[1] < (vox[1]-1))
    dist[1] = y - (float)tfl[1];
  else                                            // border values need special treatment
  {
    --tfl[1];
    dist[1] = 1.0f;
  }
  if (tfl[2] < (vox[2]-1))
    dist[2] = z - (float)tfl[2];
  else                                            // border values need special treatment
  {
    --tfl[2];
    dist[2] = 1.0f;
  }

  raw.makeCurrent(f);
  rd = raw.getData();                             // get pointer to voxel data

  // Compute pointers to neighboring voxels:
  sliceSize = vox[0] * vox[1] * getBPV();
  lineSize  = vox[0] * getBPV();
  neighbor[0] = rd + getBPV() * tfl[0] + tfl[1] * lineSize + tfl[2] * sliceSize;
  neighbor[1] = neighbor[0] + lineSize;
  neighbor[2] = neighbor[1] + getBPV();
  neighbor[3] = neighbor[0] + getBPV();
  neighbor[4] = neighbor[0] + sliceSize;
  neighbor[5] = neighbor[1] + sliceSize;
  neighbor[6] = neighbor[2] + sliceSize;
  neighbor[7] = neighbor[3] + sliceSize;

  for (size_t j=0; j<chan; ++j)
  {
    // Get neighboring voxel values:
    for (size_t i=0; i<8; ++i)
    {
      switch (bpc)
      {
        default:
        case 1: val[i] = (float)(*neighbor[i] + j); break;
        case 2: val[i] = ((float)(*neighbor[i]+j*2)) * 256.0f + (float)(*(neighbor[i]+j*2+1)); break;
        case 4: val[i] = (float)(*(neighbor[i]+j*4)); break;
      }
    }

    // Trilinearly interpolate values:
    interpolated = (int)(
      val[0] * (1.0f - dist[0]) * (1.0f - dist[1]) * (1.0f - dist[2]) +
      val[1] * (1.0f - dist[0]) * dist[1]          * (1.0f - dist[2]) +
      val[2] * dist[0]          * dist[1]          * (1.0f - dist[2]) +
      val[3] * dist[0]          * (1.0f - dist[1]) * (1.0f - dist[2]) +
      val[4] * (1.0f - dist[0]) * (1.0f - dist[1]) * dist[2] +
      val[5] * (1.0f - dist[0]) * dist[1]          * dist[2] +
      val[6] * dist[0]          * dist[1]          * dist[2] +
      val[7] * dist[0]          * (1.0f - dist[1]) * dist[2] );

    switch (bpc)
    {
      default:
      case 1: result[j]     = (uint8_t)interpolated; break;
      case 2: result[j*2]   = (uint8_t)(interpolated / 256);
      result[j*2+1] = (uint8_t)(interpolated % 256); break;
      case 4: *((float*)(result+j*4)) = float(interpolated); break;
    }
  }
}

//----------------------------------------------------------------------------
/** Draws a 3D box into the current animation frame of the dataset.
  @param p1x,p1y,p1z   box start point
  @param p2x,p2y,p2z   box end point
  @param chan          channel to draw in
  @param val           value of channel voxel, array size must equal bpc
*/
void vvVolDesc::drawBox(ssize_t p1x, ssize_t p1y, ssize_t p1z, ssize_t p2x, ssize_t p2y, ssize_t p2z, size_t chan, uint8_t* val)
{
  uint8_t* raw;
  size_t lineSize, sliceSize;

  vvDebugMsg::msg(3, "vvVolDesc::drawBox()");

  p1x = ts_clamp(p1x, ssize_t(0), vox[0]-1);
  p1y = ts_clamp(p1y, ssize_t(0), vox[1]-1);
  p1z = ts_clamp(p1z, ssize_t(0), vox[2]-1);
  p2x = ts_clamp(p2x, ssize_t(0), vox[0]-1);
  p2y = ts_clamp(p2y, ssize_t(0), vox[1]-1);
  p2z = ts_clamp(p2z, ssize_t(0), vox[2]-1);

  sliceSize = getSliceBytes();
  lineSize  = vox[0] * getBPV();
  raw = getRaw(currentFrame);
  for (ssize_t z=0; z<vox[2]; ++z)
  {
    for (ssize_t y=0; y<vox[1]; ++y)
    {
      for (ssize_t x=0; x<vox[0]; ++x)
      {
        if (x>=p1x && y>=p1y && z>=p1z &&
          x<=p2x && y<=p2y && z<=p2z)
        {
          bool bPaint = true;
          /*      for (c=0; c<bpc*chan; ++c)
                {
                  if (raw[z* sliceSize + y * lineSize + x * getBPV() + c] > 5)
                    bPaint = false;
                }*/
          if (bPaint)
          {
            for (size_t c=0; c<bpc*chan; ++c)
            {
              raw[z * sliceSize + y * lineSize + x * getBPV() + c] = val[c];
            }
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
/** Draws a 3D sphere into the current animation frame of the dataset.
  @param p1x,p1y,p1z   sphere start point
  @param radius        sphere radius
  @param chan          channel to draw in
  @param val           value of channel voxel, array size must equal bpc
*/
void vvVolDesc::drawSphere(ssize_t p1x, ssize_t p1y, ssize_t p1z, ssize_t radius, size_t chan, uint8_t* val)
{
  /*
  if (_radius != radius)
  {
    _radius = radius;
    if (_mask != NULL)
      delete _mask;
    int msize = radius * 2 + 1;
    float distance;
    float zscore;

    _mask = new int[msize * msize * msize];

  for (int z = 0; z < msize; z++)
  for (int x = 0; x < msize; x++)
  for (int y = 0; y < msize; y++)
  {
  distance = (x - radius) * (x - radius) + (y - radius) * (y - radius) + (z - radius) * (z - radius);
  zscore = distance / (radius  * radius * radius) / 0.05;
  if (radius * radius * radius> distance)
  mask[z * msize * msize + x * msize + y] = 255 * ( pow (M_E, -1.0 * zscore / 2) / sqrt(2 * M_PI));
  else
  mask[z * msize * msize + x * msize + y] = 0;
  }
  }
  */
  ssize_t xstart = ts_max(ssize_t(0), p1x - radius);
  ssize_t xend = ts_min(vox[0], p1x + radius);
  ssize_t ystart = ts_max(ssize_t(0), p1y - radius);
  ssize_t yend = ts_min(vox[1], p1y + radius);
  ssize_t zstart = ts_max(ssize_t(0), p1z - radius);
  ssize_t zend = ts_min(vox[2], p1z + radius);

  cerr << xstart << " " << xend << " " << ystart << " " << yend << " " << zstart << " " << zend << endl;

  cerr << radius << endl;
  uint8_t* raw;
  size_t lineSize, sliceSize;

  sliceSize = getSliceBytes();
  lineSize  = vox[0] * getBPV();
  raw = getRaw(currentFrame);
  for (ssize_t z = zstart; z < zend; ++z)
  {
    for (ssize_t y = ystart; y < yend; ++y)
    {
      for (ssize_t x = xstart; x < xend; ++x)
      {
        ssize_t dist = (ssize_t)sqrt(float((x - p1x) * (x - p1x) + (y - p1y) * (y - p1y) + (z - p1z) * (z - p1z)));

        if (radius > dist)
          for (size_t c=0; c<bpc*chan; ++c)
        {
          raw[z * sliceSize + y * lineSize + x * getBPV() + c] = val[c] * (1 - dist / radius);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
/** Draws a 3D line into the current animation frame of the dataset.
  @param p1x,p1y,p1z   line start point
  @param p2x,p2y,p2z   line end point
  @param val           value of line voxels, array size must equal bpc * chan
*/
void vvVolDesc::drawLine(ssize_t p1x, ssize_t p1y, ssize_t p1z, ssize_t p2x, ssize_t p2y, ssize_t p2z, uint8_t* val)
{
  uint8_t* raw;

  vvDebugMsg::msg(3, "vvVolDesc::drawLine()");

  raw = getRaw(currentFrame);
  vvToolshed::draw3DLine(p1x, p1y, p1z, p2x, p2y, p2z, val,
    raw, getBPV(), vox[0], vox[1], vox[2]);
}

//----------------------------------------------------------------------------
/** Draws boundary lines around the volume. The lines are one voxel thick.
  @param color boundary color array, size must equal bpv!
  @param frame frame to draw boundaries for (-1 for all frames)
*/
void vvVolDesc::drawBoundaries(uchar* color, int frame)
{
  uint8_t* raw;
  int f;                                          // frame counter
  int i;
  int from, to;
  uchar lines[12][2][3] =
  {
    {
      {
        0, 0, 0
      }
      ,
      {
        1, 0, 0
      }
    },
    {
      {
        1, 0, 0
      }
      ,
      {
        1, 0, 1
      }
    },
    {
      {
        0, 0, 1
      }
      ,
      {
        1, 0, 1
      }
    },
    {
      {
        0, 0, 0
      }
      ,
      {
        0, 0, 1
      }
    },
    {
      {
        0, 1, 0
      }
      ,
      {
        1, 1, 0
      }
    },
    {
      {
        1, 1, 0
      }
      ,
      {
        1, 1, 1
      }
    },
    {
      {
        0, 1, 1
      }
      ,
      {
        1, 1, 1
      }
    },
    {
      {
        0, 1, 0
      }
      ,
      {
        0, 1, 1
      }
    },
    {
      {
        0, 0, 0
      }
      ,
      {
        0, 1, 0
      }
    },
    {
      {
        1, 0, 0
      }
      ,
      {
        1, 1, 0
      }
    },
    {
      {
        1, 0, 1
      }
      ,
      {
        1, 1, 1
      }
    },
    {
      {
        0, 0, 1
      }
      ,
      {
        0, 1, 1
      }
    },
  };

  vvDebugMsg::msg(3, "vvVolDesc::drawBoundaries()");

  if (frame<0)
  {
    from = 0;
    to = frames-1;
  }
  else
  {
    from = frame;
    to = frame;
  }
  for (f=from; f<=to; ++f)
  {
    raw = getRaw(f);
    for (i=0; i<12; ++i)
    {
      vvToolshed::draw3DLine(lines[i][0][0] * vox[0], lines[i][0][1] * vox[1], lines[i][0][2] * vox[2],
        lines[i][1][0] * vox[0], lines[i][1][1] * vox[1], lines[i][1][2] * vox[2],
        color, raw, getBPV(), vox[0], vox[1], vox[2]);
    }
  }
}

//----------------------------------------------------------------------------
/** Serializes all volume attributes to a memory buffer.
  Serialization is performed for volume size, number of voxels in each axis,
  number of time steps, real voxel sizes, etc.<BR>
  This serialization does NOT include the voxel data and the transfer functions!<BR>
  Since the serialization buffer must be allocated before calling the function,
  its size can be found by calling the function with the NULL parameter or no
  parameter at all. Here's an example:<PRE>
  size_t num_bytes = serializeAttributes();
  uchar* buffer = new uchar[num_bytes];
  serializeAttributes(buffer);
  </PRE><BR>
The @see deserializeAttributes command can be used to deserialize the
buffer values.<P>
Here is the exact description of the buffer values in the serialized
buffer:
<PRE>
Length          Data Type        VolDesc Attribute
---------------------------------------------------------------
3 x 4 bytes     unsigned int     vox[0..2]
4 bytes         unsigned int     frames
1 byte          unsigned char    bpv (bytes per voxel)
3 x 4 bytes     float            dist[0..2]
4 bytes         float            dt
2 x 4 bytes     float            realMin, realMax
3 x 4 bytes     float            pos
1 byte          unsigned char    storage type
</PRE>
@param buffer pointer to _allocated_ memory for serialized attributes
@return number of bytes required for serialization buffer
*/
size_t vvVolDesc::serializeAttributes(uint8_t* buffer) const
{
  uint8_t* ptr;                                     // pointer to current serialization buffer element

  vvDebugMsg::msg(3, "vvVolDesc::serializeAttributes()");

  if (buffer != NULL)
  {
    ptr = buffer;
    ptr += virvo::serialization::write32 (ptr, vox[0]);
    ptr += virvo::serialization::write32 (ptr, vox[1]);
    ptr += virvo::serialization::write32 (ptr, vox[2]);
    ptr += virvo::serialization::write32 (ptr, frames);
    ptr += virvo::serialization::write8    (ptr, uint8_t(bpc));
    ptr += virvo::serialization::writeFloat(ptr, dist[0]);
    ptr += virvo::serialization::writeFloat(ptr, dist[1]);
    ptr += virvo::serialization::writeFloat(ptr, dist[2]);
    ptr += virvo::serialization::writeFloat(ptr, dt);
    ptr += virvo::serialization::writeFloat(ptr, real[0]);
    ptr += virvo::serialization::writeFloat(ptr, real[1]);
    ptr += virvo::serialization::writeFloat(ptr, pos[0]);
    ptr += virvo::serialization::writeFloat(ptr, pos[1]);
    ptr += virvo::serialization::writeFloat(ptr, pos[2]);
    ptr += virvo::serialization::write8    (ptr, uint8_t(chan));
    assert(ptr - buffer == SERIAL_ATTRIB_SIZE);
  }
  return SERIAL_ATTRIB_SIZE;
}

//----------------------------------------------------------------------------
/** Deserializes all volume attributes from a memory buffer.
  The @see serializeAttributes command can be used to serialize the
  attributes.
  @param buffer   pointer to _allocated_ memory for serialized attributes
  @param bufSize  size of buffer [bytes]. Values smaller than the default
                  size are allowed and only fill the values up to the
                  passed value. The remaining values will be set to default values.
*/
void vvVolDesc::deserializeAttributes(uint8_t* buffer, size_t bufSize)
{
  vvDebugMsg::msg(3, "vvVolDesc::deserializeAttributes()");
  assert(buffer!=NULL);

  // Set default values for all serializable attributes:
  setDefaults();

  uint8_t* ptr = buffer;

  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    vox[0] = virvo::serialization::read32(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    vox[1] = virvo::serialization::read32(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    vox[2] = virvo::serialization::read32(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    frames = virvo::serialization::read32(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+1 - buffer) <= bufSize)
    bpc = virvo::serialization::read8(ptr);
  else return;
  ptr += 1;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    dist[0] = virvo::serialization::readFloat(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    dist[1]  = virvo::serialization::readFloat(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    dist[2] = virvo::serialization::readFloat(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    dt = virvo::serialization::readFloat(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    real[0] = virvo::serialization::readFloat(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    real[1] = virvo::serialization::readFloat(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    pos[0] = virvo::serialization::readFloat(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    pos[1] = virvo::serialization::readFloat(ptr);
  else return;
  ptr += 4;
  assert(ptr + 4 - buffer >= 0);
  if (size_t(ptr+4 - buffer) <= bufSize)
    pos[2] = virvo::serialization::readFloat(ptr);
  else return;
  ptr += 4;
  assert(ptr + 1 - buffer >= 0);
  if (size_t(ptr+1 - buffer) <= bufSize)
    chan = virvo::serialization::read8(ptr);
  else return;
  ptr += 1;
}

//----------------------------------------------------------------------------
/** Set the data of one slice to new values.
  @param newData new voxel data for one specific slice
                 (bpv*width*height voxels expected). The data must be arranged
                 in the same way as the volume data, i.e. all bytes of the voxel
                 must be in a row. The data will be copied to the volume dataset,
                 so it must be freed by the _caller_!
  @param slice   slice index (first slice = 0 = default)
  @param frame   frame index (first frame = 0 = default)
*/
void vvVolDesc::setSliceData(uint8_t* newData, int slice, int frame)
{
  uint8_t* dst;                                   // pointer to beginning of slice
  size_t sliceSize;                               // shortcut for speed

  if (frames>0 && vox[2]>0)                       // make sure at least one slice is stored
  {
    sliceSize = getSliceBytes();
    dst = getRaw(frame) + slice * sliceSize;
    memcpy(dst, newData, sliceSize);
  }
}

//----------------------------------------------------------------------------
/** Extract a slice from a volume. Z direction slices are drawn just as they
  are in the volume. For X direction slices, the height is the same as for
  Z slices, and the width is the depth of the volume. For Y direction slices,
  the width is the same as for the volume, and the height is the depth of
  the volume. The slices are drawn as they are seen from the positive end of
  the respective axis.
  @param frame  animation frame [0..frames-1]; -1 for current frame
  @param axis   axis for which to generate slice data
  @param slice  slice index to create, relative to slicing axis (>=0)
  @param dst    _allocated_ space for sliceWidth * sliceHeight * bpc * chan bytes
*/
void vvVolDesc::extractSliceData(int frame, vvVecmath::AxisType axis, size_t slice, uint8_t* dst)
{
  uint8_t* raw;                                   // raw volume data of current frame
  size_t sliceSize;                               // bytes per volume slice (z-axis view)
  size_t lineSize;                                // bytes per voxel line

  sliceSize = getSliceBytes();
  lineSize  = vox[0] * getBPV();
  raw = getRaw(frame);
  switch (axis)
  {
    case 0:                                       // x axis
      for (ssize_t j=0; j<vox[1]; ++j)
        for (ssize_t i=0; i<vox[2]; ++i)
      {
        memcpy(dst, raw + i * sliceSize + j * lineSize + (vox[0] - slice - 1) * getBPV(), getBPV());
        dst += getBPV();
      }
      break;
    case 1:                                       // y axis
      for (ssize_t i=0; i<vox[2]; ++i)
        memcpy(dst + i * lineSize, raw + (vox[2] - i - 1) * sliceSize + slice * lineSize, lineSize);
      break;
    case 2:                                       // z axis
      memcpy(dst, raw + slice * sliceSize, sliceSize);
      break;
    default: assert(0); break;
  }
}

//----------------------------------------------------------------------------
/** @return volume size when looking from a specific axis
  @param axis viewing axis
*/
void vvVolDesc::getVolumeSize(vvVecmath::AxisType axis, size_t& width, size_t& height, size_t& slices)
{
  switch(axis)
  {
    case vvVecmath::X_AXIS: width = vox[2]; height = vox[1]; slices = vox[0]; return;
    case vvVecmath::Y_AXIS: width = vox[0]; height = vox[2]; slices = vox[1]; return;
    case vvVecmath::Z_AXIS: width = vox[0]; height = vox[1]; slices = vox[2]; return;
    default: assert(0); width = 0; height = 0; slices = 0; return;
  }
}

//----------------------------------------------------------------------------
/** Renders a slice into an RGB data buffer.
  @param frame  animation frame [0..frames-1]; -1 for current frame
  @param axis   axis for which to generate slice data
  @param slice  slice index to create, relative to slicing axis (>=0)
  @param dst    _allocated_ space for sliceWidth * sliceHeight * 3 bytes;
                get width and height via getVolumeSize
*/
void vvVolDesc::makeSliceImage(int frame, vvVecmath::AxisType axis, size_t slice, uint8_t* dst)
{
  uint8_t* sliceData;
  vvColor col;
  float voxelVal = 0.f;
  size_t sliceBytes;
  size_t width, height, slices;
  size_t pixel;
  size_t srcOffset, dstOffset;

  getVolumeSize(axis, width, height, slices);
  sliceBytes = width * height * bpc * chan;
  sliceData = new uint8_t[sliceBytes];
  extractSliceData(frame, axis, slice, sliceData);
  memset(dst, 0, width * height * 3);

  for(pixel=0; pixel<width*height; ++pixel)
  {
    for (size_t c=0; c<ts_min(chan, size_t(3)); ++c)
    {
      dstOffset = pixel * 3 + c;
      srcOffset = pixel * bpc * chan + bpc * c;

      switch(bpc)
      {
        case 1: voxelVal = float(sliceData[srcOffset]) / 255.0f; break;
#ifdef BOOST_LITTLE_ENDIAN
        case 2: voxelVal = float(int(sliceData[srcOffset + 1]) * 256 + int(sliceData[srcOffset])) / 65535.0f; break;
#else
        case 2: voxelVal = float(int(sliceData[srcOffset]) * 256 + int(sliceData[srcOffset + 1])) / 65535.0f; break;
#endif
        case 4: voxelVal = ((*((float*)(sliceData+srcOffset))) - real[0]) / (real[1] - real[0]); break;
        default: assert(0); break;
      }
      if (chan==1)
      {
        col = tf.computeColor(voxelVal);
        dst[dstOffset]     = int(col[0] * 255.0f);
        dst[dstOffset + 1] = int(col[1] * 255.0f);
        dst[dstOffset + 2] = int(col[2] * 255.0f);
      }
      else                                        // 2-3 channels: use as RGB
      {
        dst[dstOffset] = int(voxelVal * 255.0f);
      }
    }
  }
}

//----------------------------------------------------------------------------
/** Corrects an image with interlaced slices. The first slice will remain
  the first slice, the second slice will be taken from halfway into the
  dataset.
*/
void vvVolDesc::deinterlace()
{
  uint8_t* rd;
  uint8_t* volBuf;                                 // temporary buffer for voxel data of one time step
  uint8_t* dst;                                   // destination pointer
  uint8_t* src;                                   // source pointer
  size_t sliceSize;
  size_t frameSize;

  vvDebugMsg::msg(2, "vvVolDesc::deinterlace()");

  sliceSize = getSliceBytes();
  frameSize = getFrameBytes();
  volBuf = new uint8_t[frameSize];
  assert(volBuf);
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    memcpy(volBuf, rd, frameSize);                // make backup copy of volume
    for (ssize_t z=0; z<vox[2]; ++z)
    {
      dst = rd + z * sliceSize;
      if ((z % 2) == 0)                           // even slice number?
        src = volBuf + (z/2) * sliceSize;
      else
        src = volBuf + (z/2 + (vox[2]+1)/2) * sliceSize;

      // Swap source and destination slice:
      memcpy((void*)dst, (void*)src, sliceSize);
    }
    raw.next();
  }
  delete[] volBuf;
}

//----------------------------------------------------------------------------
/** Find the minimum and maximum scalar value.
  @param channel data channel to search
  @param scalarMin,scalarMax  minimum and maximum scalar values in volume animation
*/
void vvVolDesc::findMinMax(size_t channel, float& scalarMin, float& scalarMax)
{
  (void)channel;
  int mi, ma;
  float fMin, fMax;

  vvDebugMsg::msg(2, "vvVolDesc::findMinMax()");

  switch(bpc)
  {
    case 1:
      scalarMin = 255.0f;
      scalarMax = 0.0f;
      break;
    case 2:
      scalarMin = 65535.0f;
      scalarMax = 0.0f;
      break;
    case 4:
      scalarMin =  VV_FLT_MAX;
      scalarMax = -VV_FLT_MAX;
      break;
    default: assert(0); break;
  }

  // TODO: make search channel dependent
  for (size_t f=0; f<frames; ++f)
  {
    switch(bpc)
    {
      case 1: vvToolshed::getMinMax(getRaw(f), getFrameBytes(), &mi, &ma);
        fMin = float(mi);
        fMax = float(ma);
        break;
      case 2: vvToolshed::getMinMax16bitBE(getRaw(f), getFrameVoxels(), &mi, &ma);
        fMin = float(mi);
        fMax = float(ma);
        break;
      case 4:
        vvToolshed::getMinMax((float*)getRaw(f), getFrameVoxels(), &fMin, &fMax);
        break;
      default: assert(0); break;
    }
    if (fMin < scalarMin) scalarMin = fMin;
    if (fMax > scalarMax) scalarMax = fMax;
  }
}

//----------------------------------------------------------------------------
/** Find the value which splits the number of values smaller or greater than
  it into the ratio given by threshold. Example: if threshold is 0.05, then
  the returned value X says that 5% of the data values are smaller than X,
  and 95% are greater than X.
  @param frame animation frame to work on, first frame is 0. -1 for all frames
  @param channel data channel to work on
  @param threshold  threshold value for data range clamping [0..1]
*/
float vvVolDesc::findClampValue(int frame, size_t channel, float threshold)
{
  int* hist;
  int buckets[1] = {1000};
  float fMin, fMax;
  float clampVal = 0.0f;
  size_t voxelCount=0;
  int i;

  vvDebugMsg::msg(2, "vvVolDesc::findClampValue()");

  if (threshold<0.0f || threshold>1.0f) cerr << "Warning: threshold clamped to 0..1" << endl;
  threshold = ts_clamp(threshold, 0.0f, 1.0f);

  findMinMax(channel, fMin, fMax);
  real[0] = fMin;
  real[1] = fMax;
  size_t frameVoxels = getFrameVoxels();
  size_t thresholdVoxelCount = size_t(float(frameVoxels) * threshold);
  hist = new int[buckets[0]];
  makeHistogram(frame, channel, 1, buckets, hist, real[0], real[1]);
  for (i=0; i<buckets[0]; ++i)
  {
    if (voxelCount >= thresholdVoxelCount)
    {
      clampVal = (float(i) / (buckets[0] - 1)) * (fMax - fMin) + fMin;
      break;
    }
    voxelCount += hist[i];
  }
  if (i==buckets[0] && clampVal==0.0f)  // for loop didn't break
  {
    clampVal = fMax;
  }
  delete[] hist;

  return clampVal;
}

//----------------------------------------------------------------------------
/** Find the number of voxels with a specific value.<br>
  For multi-modal volumes, all channels must be equal to that value to count.
  @param frame frame index to look at (first frame = 0)
  @param val   value to look for
*/
int vvVolDesc::findNumValue(int frame, float val)
{
  uint8_t* raw;
  int num = 0;
  int sval;
  bool allEqual;

  vvDebugMsg::msg(2, "vvVolDesc::findNumValue()");

  size_t frameVoxels = getFrameVoxels();

  // Search volume:
  raw = getRaw(frame);
  for (size_t i=0; i<frameVoxels; ++i)
  {
    allEqual = true;
    for (size_t m=0; m<chan; ++m)
    {
      switch (bpc)
      {
        case 1:
          if (raw[i] != uint8_t(val)) allEqual = false;
          break;
        case 2:
          sval = int(raw[i]) * 256 + raw[i+1];
          if (sval != int(val)) allEqual = false;
          break;
        case 4:
          if (*((float*)(raw+i)) != val) allEqual = false;
          break;
      }
    }
    if (allEqual) ++num;
  }
  return num;
}

//----------------------------------------------------------------------------
/** Find the number of different data values used in a dataset.
  @return -1 if data type is float
*/
int vvVolDesc::findNumUsed(size_t channel)
{
  bool* used;                                     // true = scalar value occurs in array
  uint8_t* raw;
  size_t numValues;
  int numUsed = 0;
  int value;

  vvDebugMsg::msg(2, "vvVolDesc::findNumUsed()");

  if (bpc>=3) return -1;     // doesn't work with floats

  size_t frameVoxels = getFrameVoxels();
  numValues = (bpc==2) ? 65536 : 256;
  used = new bool[numValues];

  // Initialize occurrence array:
  for (size_t i=0; i<numValues; ++i)
  {
    used[i] = false;
  }

  // Fill occurrence array:
  for (size_t f=0; f<frames; ++f)
  {
    raw = getRaw(f);
    for (size_t i=0; i<frameVoxels; ++i)
    {
      switch (bpc)
      {
        case 1:
          used[raw[i*chan+channel]] = true;
          break;
        case 2:
          value = (int(raw[i*2*chan+channel]) << 8) | int(raw[i*2*chan+channel+1]);
          used[value] = true;
          break;
      }
    }
  }

  // Count number of 'true' entries in occurrence array:
  for (size_t i=0; i<numValues; ++i)
  {
    if (used[i]==true) ++numUsed;
  }

  delete[] used;
  return numUsed;
}

//----------------------------------------------------------------------------
/** Find the number of transparent data values in a dataset.
  If there is no transfer function defined in the VD,
  a linear ramp is assumed.
  @param frame frame index to look at (first frame = 0)
*/
int vvVolDesc::findNumTransparent(int frame)
{
  float* rgba = NULL;
  uint8_t* raw;
  int numTransparent = 0;
  int lutEntries = 0;
  int scalar;
  bool noTF;                                      // true = no TF present in file

  vvDebugMsg::msg(2, "vvVolDesc::findNumTransparent()");

  if (bpc==4) return 0;                           // TODO: implement for floats

  size_t frameSize = getFrameBytes();

  noTF = tf._widgets.empty();

  if (!noTF)
  {
    switch(bpc)
    {
      case 1: lutEntries = 256;   break;
      case 2: lutEntries = 65536; break;
    }

    rgba = new float[4 * lutEntries];

    // Generate arrays from pins:
    tf.computeTFTexture(lutEntries, 1, 1, rgba, real[0], real[1]);
  }

  // Search volume:
  raw = getRaw(frame);
  switch (bpc)
  {
    case 1:
    case 2:
    case 4:
      for (size_t i=bpc-1; i<frameSize; i += getBPV())
      {
        if (bpc==2) scalar = (int(raw[i-1]) << 8) | int(raw[i]);
        else scalar = raw[i];
        if (noTF)
        {
          if (scalar==0) ++numTransparent;
        }
        else
        {
          if (rgba[scalar * 4 + 3]==0.0f) ++numTransparent;
        }
      }
      break;
  }

  if (!noTF) delete[] rgba;

  return numTransparent;
}

//----------------------------------------------------------------------------
/** Calculate the mean of the values in a dataset.
  @param frame frame index to look at (first frame = 0)
*/
float vvVolDesc::calculateMean(int frame)
{
  uint8_t* raw;
  float scalar = 0.f;
  float mean;
  double sum = 0.0;

  vvDebugMsg::msg(2, "vvVolDesc::calculateMean()");

  size_t frameSize = getFrameBytes();
  size_t bpv = getBPV();

  // Search volume:
  raw = getRaw(frame);
  for (size_t i=0; i<frameSize; i += bpv)
  {
    switch (bpc)
    {
      case 1:
        scalar = float(raw[i]);
        break;
      case 2:
        scalar = float((int(raw[i]) << 8) | int(raw[i+1]));
        break;
      case 4:
        scalar = *((float*)(raw+i));
        break;
      default: assert(0); break;
    }
    sum += scalar;
  }
  mean = float(sum / double(getFrameVoxels()));
  return mean;
}

//----------------------------------------------------------------------------
/** Calculate mean, variance, and standard deviation of the values in a dataset.
  @param frame frame index to look at (first frame = 0)
  @param chan channel to look at
  @return mean, variance, stdev
*/
void vvVolDesc::calculateDistribution(int frame, size_t chan, float& mean, float& variance, float& stdev)
{
  uint8_t* raw;
  double sumSquares = 0.0;
  float scalar = 0.f;
  float diff;

  vvDebugMsg::msg(2, "vvVolDesc::calculateDistribution()");

  size_t frameSize = getFrameBytes();
  size_t bpv = getBPV();
  mean = calculateMean(frame);

  // Search volume:
  raw = getRaw(frame);
  for (size_t i=chan*bpc; i<frameSize; i += bpv)
  {
    switch (bpc)
    {
      case 1:
        scalar = float(raw[i]);
        break;
      case 2:
        scalar = float((int(raw[i]) << 8) | int(raw[i+1]));
        break;
      case 4:
        scalar = *((float*)(raw+i));
        break;
      default: assert(0); break;
    }
    diff = scalar - mean;
    sumSquares += diff * diff;
  }
  variance = float(sumSquares / double(getFrameVoxels()));
  stdev = sqrtf(variance);
}

//----------------------------------------------------------------------------
/**
  Expand the data range to occupy the entire value range. This is useful to take
  advantage of the entire possible value range. For example, if the scanned
  16 bit data occupy only values between 50 and 1000, they will be mapped to
  a range of 0 to 65535.
  @param verbose true = display progress
*/
void vvVolDesc::expandDataRange(bool verbose)
{
  float smin, smax;                               // scalar minimum and maximum

  vvDebugMsg::msg(2, "vvVolDesc::expandDataRange()");

  findMinMax(0, smin, smax);
  zoomDataRange(-1, int(smin), int(smax), verbose);
}

//----------------------------------------------------------------------------
/** Zoom in on a subset of the value range for a more useful distribution
    of the data range of interest. Floating point volumes are not affected
    by this function.
  @param channel channel to zoom values of (-1 for all channels)
  @param low,high bottom and top limit of data range to zoom to [scalar value]
  @param verbose true = display progress
*/
void vvVolDesc::zoomDataRange(int channel, int low, int high, bool verbose)
{
  uint8_t* raw;                                   // raw volume data
  size_t frameSize;                               // values per frame
  int ival;
  float fmin, fmax, fval, frange;

  vvDebugMsg::msg(2, "vvVolDesc::zoomDataRange()");

  if (bpc>2) return;                              // nothing to be done

  frameSize = getFrameVoxels();
  fmin = float(low);
  fmax = float(high);
  frange = fmax - fmin;

  // Compute new real world range:
  if (chan==1 || channel==-1)                     // effect on real range undefined if only one channel is zoomed
  {
    int irange = (bpc==2) ? 65535 : 255;
    real[0] = fmin / float(irange) * (real[1] - real[0]) + real[0];
    real[1] = fmax / float(irange) * (real[1] - real[0]) + real[0];
  }

  // Perform the actual expansion:
  if (verbose) vvToolshed::initProgress(frames);
  for (size_t f=0; f<frames; ++f)
  {
    raw = getRaw(f);
    for (size_t i=0; i<frameSize; ++i)
    {
      for (size_t c=0; c<chan; ++c)
      {
        if (c==static_cast<size_t>(channel) || channel<0)
        {
          switch (bpc)
          {
            case 1:
              fval = float(*raw);
              ival = int(255.0f * (fval-fmin) / frange);
              ival = ts_clamp(ival, 0, 255);
              *raw = uint8_t(ival);
              break;
            case 2:
              ival = (int(*raw) << 8) | int(*(raw+1));
              fval = float(ival);
              ival = int(65535.0f * (fval - fmin) / frange);
              ival = ts_clamp(ival, 0, 65535);
              *raw = uint8_t(ival >> 8);
              *(raw+1) = uint8_t(ival & 0xFF);
              break;
            default: assert(0); break;
          }
        }
        raw += bpc;
      }
    }
    if (verbose) vvToolshed::printProgress(f);
  }
  if (verbose) cerr << endl;
}

//----------------------------------------------------------------------------
/** Blend two volumes together. Both volumes must have the same size and
  the same data type. If the number of time steps is different,
  the sequence with less steps will be repeated. The result will have
  the same number of frames as the source volume.
  During blending, each component of each voxel is blended with the
  corresponding component of the other volume individually. The
  result is defined by the blend method.
  @param blendVD volume to blend
  @param method  blending method: 0=average, 1=max, 2=min
  @param verbose true = display progress
*/
void vvVolDesc::blend(vvVolDesc* blendVD, int method, bool verbose)
{
  uint8_t* raw;                                   // raw volume data
  uint8_t* rawBlend;                              // raw volume data of file to blend
  size_t frameSize;                               // bytes per frame
  size_t fBlend;
  float val1, val2;
  float blended;                                  // result from blending operation

  vvDebugMsg::msg(2, "vvVolDesc::blend()");

  if (bpc != blendVD->bpc || chan != blendVD->chan || vox[0] != blendVD->vox[0] ||
    vox[1] != blendVD->vox[1] || vox[2] != blendVD->vox[2] ||
    dist[0] != blendVD->dist[0] || dist[1] != blendVD->dist[1] ||
    dist[2] != blendVD->dist[2] || dt != blendVD->dt)
  {
    cerr << "Cannot blend: volumes to blend must have same size, number of channels, bytes per channel, and voxel distances" << endl;
    return;
  }

  frameSize = getFrameBytes();
  fBlend = 0;

  if (verbose) vvToolshed::initProgress(frames);
  for (size_t f=0; f<frames; ++f)
  {
    raw = getRaw(f);
    rawBlend = blendVD->getRaw(fBlend);
    for (size_t i=0; i<frameSize; ++i)                   // step through all voxel bytes
    {
      switch(bpc)
      {
        case 1:
          val1 = float(raw[i]);
          val2 = float(rawBlend[i]);
          break;
        case 2:
          val1 = float(int(raw[i]) * 256 + int(raw[i+1]));
          val2 = float(int(rawBlend[i]) * 256 + int(rawBlend[i+1]));
          break;
        case 4:
          val1 = *((float*)(raw+i));
          val2 = *((float*)(rawBlend+i));
          break;
        default: assert(0); val1 = val2 = 0.0f; break;
      }
      switch (method)
      {
        case 0:  blended = (val1 + val2) / 2.0f; break;
        case 1:  blended = ts_max(val1, val2); break;
        case 2:  blended = ts_min(val1, val2); break;
        default: blended = val1; break;
      }
      switch(bpc)
      {
        case 1:
          raw[i] = uint8_t(blended);
          break;
        case 2:
          raw[i]   = uint8_t(int(blended) >> 8);
          raw[i+1] = uint8_t(int(blended) & 0xff);
          ++i;                                    // skip second byte
          break;
        case 4:
          *((float*)(raw+i)) = blended;
          i += 3;                                 // skip rest of float bytes
          break;
      }
    }
    if (verbose) vvToolshed::printProgress(f);
    ++fBlend;
    if (fBlend>=blendVD->frames) fBlend = 0;
  }
  if (verbose) cerr << endl;
}

//----------------------------------------------------------------------------
/** Swap the values of two channels in every voxel and every
  animation frame.
  @param ch0,ch1 channel indices to swap (start with 0)
  @param verbose true = print progress info
*/
void vvVolDesc::swapChannels(size_t ch0, size_t ch1, bool verbose)
{
  uint8_t* rd;
  size_t sliceSize;
  size_t rowOffset, sliceOffset, voxelOffset;
  uint8_t buffer[4];                                // buffer for one channel value while swapping
  uint8_t* ptr0;
  uint8_t* ptr1;

  vvDebugMsg::msg(2, "vvVolDesc::swapChannels()");
  if (ch0==ch1) return;                           // this was easy!
  assert(bpc<=4);                                 // determines buffer size

  sliceSize = getSliceBytes();
  raw.first();
  if (verbose) vvToolshed::initProgress(frames * vox[2]);
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    raw.next();
    for (ssize_t z=0; z<vox[2]; ++z)
    {
      sliceOffset = z * sliceSize;
      for (ssize_t y=0; y<vox[1]; ++y)
      {
        rowOffset = sliceOffset + y * vox[0] * getBPV();
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          voxelOffset = x * getBPV() + rowOffset;

          ptr0 = rd + voxelOffset + ch0 * bpc;
          ptr1 = rd + voxelOffset + ch1 * bpc;
          memcpy(buffer, ptr0, bpc);              // copy channel 0 to buffer
          memcpy(ptr0, ptr1, bpc);                // copy channel 1 to channel 0
          memcpy(ptr1, buffer, bpc);              // copy buffer to channel 1
        }
      }
      if (verbose) vvToolshed::printProgress(f * vox[2] + z);
    }
  }
  if (verbose) cerr << endl;

  // Adjust channel names:
  std::swap(channelNames[ch0], channelNames[ch1]);
}

//----------------------------------------------------------------------------
/** Extract a color of an RGB data set and make a 4th channel out of it.
  For the source data set bpc must be 1 and chan must be 3. The data set
  will be bpc=1 and chan=4.
  @param weights weights to identify color to move to 4th channel
  @param verbose true = print progress info
*/
void vvVolDesc::extractChannel(float weights[3], bool verbose)
{
  uint8_t* newRaw;
  uint8_t* rd;
  uint8_t* src;
  uint8_t* dst;
  float val, testval;
  size_t newSliceSize;
  bool is4th;

  vvDebugMsg::msg(2, "vvVolDesc::extractChannel()");

  // Verify input parameters:
  assert(bpc==1 && chan==3);

  newSliceSize = vox[0] * vox[1] * 4;
  if (verbose) vvToolshed::initProgress(vox[2] * frames);

  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    newRaw = new uint8_t[newSliceSize * vox[2]];
    src = rd;
    dst = newRaw;
    for (ssize_t z=0; z<vox[2]; ++z)
    {
      for (ssize_t y=0; y<vox[1]; ++y)
      {
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          // Determine whether voxel belongs to 4th channel:
          val = -1.0f;                            // init
          is4th = true;
          for (int i=0; i<3; ++i)
          {
            if (weights[i] > 0.0f)
            {
              testval = float(src[i]) / weights[i];
              if (val==-1.0f) val = testval;
                                                  // allow tolerance of 0.5 to make up for roundoff errors at time of data generation
              else if (testval<val-0.1f || testval>val+0.1f) is4th = false;
            }
          }
          if (val<0.0f) is4th = false;            // all components are zero

          if (is4th)
          {
            memset(dst, 0, 3);
            dst[3] = uint8_t(val);
          }
          else                                    // not the 4th channel: copy voxel to new volume
          {
            memcpy(dst, src, 3);
            dst[3] = uint8_t(0);
          }
          src += 3;
          dst += 4;
        }
      }
      if (verbose) vvToolshed::printProgress(z + vox[2] * f);
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    raw.next();
  }
  chan = 4;

  // Adjust channel names:
  channelNames.push_back("");
}

//----------------------------------------------------------------------------
/** Computes a volume data set.
  @param algorithm algorithm to use in computation
  @param x,y,z size of volume [voxels]
*/
void vvVolDesc::computeVolume(int algorithm, size_t vx, size_t vy, size_t vz)
{
  size_t f;                                       // counters
  uint8_t* rd;                                    // raw volume data

  vvDebugMsg::msg(1, "vvFileIO::computeDefaultVolume()");

  vox[0] = vx;
  vox[1] = vy;
  vox[2] = vz;
  frames = 0;
  bpc    = 1;
  chan   = 1;
  dt     = 0.1f;

  if (getFilename()==NULL || strlen(getFilename())==0)
  {
    setFilename("default.xvf");
  }

  switch(algorithm)
  {
    case 0:                                       // default startup data set
      for (f=0; f<8; ++f)
      {
        rd = new uint8_t[getFrameBytes()];
        for (ssize_t z=0; z<vox[2]; ++z)
          for (ssize_t y=0; y<vox[1]; ++y)
            for (ssize_t x=0; x<vox[0]; ++x)
            {
              rd[x + y * vox[0] + z * getSliceBytes()] = (uint8_t)((x*(y+1)*(z+1)*(f+1)) % 256);
            }
      addFrame(rd, vvVolDesc ::ARRAY_DELETE);
      ++frames;
    }
    break;
    case 1:                                       // top=max, bottom=min
      rd = new uint8_t[getFrameBytes()];
      for (ssize_t z=0; z<vox[2]; ++z)
        for (ssize_t y=0; y<vox[1]; ++y)
          for (ssize_t x=0; x<vox[0]; ++x)
          {
            int color;
                                                  // empty boundary
            if (x<2 || y<2 || z<2 || x>vox[0]-3 || y>vox[1]-3 || z>vox[2]-3) color = 0;
            else if (y<vox[1]/2) color = 255;
            else color = 127;
            rd[x + y * vox[0] + z * vox[0] * vox[1]] = color;
          }
    addFrame(rd, vvVolDesc::ARRAY_DELETE);
    ++frames;
    break;
    case 2:                                       // 4-channel test data set
      chan = 4;
      rd = new uint8_t[getFrameBytes()];
      for (ssize_t z=0; z<vox[2]; ++z)
        for (ssize_t y=0; y<vox[1]; ++y)
          for (ssize_t x=0; x<vox[0]; ++x)
          {
            rd[chan * (x + y * vox[0] + z * vox[0] * vox[1])]     = (y < vox[1]/4) ? 255 : 0;
            rd[chan * (x + y * vox[0] + z * vox[0] * vox[1]) + 1] = (y < vox[1]/2 && y >= vox[1]/4) ? 255 : 0;
            rd[chan * (x + y * vox[0] + z * vox[0] * vox[1]) + 2] = (y >= vox[1]/2 && y < vox[1]*3/4) ? 255 : 0;
            rd[chan * (x + y * vox[0] + z * vox[0] * vox[1]) + 3] = (y >= vox[1]*3/4) ? 255 : 0;
          }
    addFrame(rd, vvVolDesc::ARRAY_DELETE);
    ++frames;
    break;
    case 3:                                       // checkercubes
    {
      const int numCubes = 4;
      int value;
      int d[3];
      for (size_t i=0; i<3; ++i) d[i] = vox[i] / ((2 * numCubes) + 1);
      rd = new uint8_t[getFrameBytes()];
      for (ssize_t z=0; z<vox[2]; ++z)
        for (ssize_t y=0; y<vox[1]; ++y)
          for (ssize_t x=0; x<vox[0]; ++x)
          {
            if ((x / d[0]) % 2 == 1 &&
              (y / d[1]) % 2 == 1 &&
              (z / d[2]) % 2 == 1) value = 255;
            else value = 0;
            rd[x + y * vox[0] + z * vox[0] * vox[1]] = value;
          }
      addFrame(rd, vvVolDesc::ARRAY_DELETE);
      ++frames;
      break;
    }
    case 4:                                       // float test data set
      bpc = 4;
      rd = new uint8_t[getFrameBytes()];
      for (ssize_t z=0; z<vox[2]; ++z)
        for (ssize_t y=0; y<vox[1]; ++y)
          for (ssize_t x=0; x<vox[0]; ++x)
          {
            *((float*)(rd + bpc * (x + y * vox[0] + z * vox[0] * vox[1]))) = float(x + y + z);
          }
    addFrame(rd, vvVolDesc::ARRAY_DELETE);
    ++frames;
    break;
    default: assert(0); break;
  }
}

vvVector3 vvVolDesc::getSize() const
{
  vvVector3 size(dist[0] * float(vox[0]) * _scale,
    dist[1] * float(vox[1]) * _scale,
    dist[2] * float(vox[2]) * _scale);
  return size;
}

void vvVolDesc::setDist(float x, float y, float z)
{
  dist[0] = x;
  dist[1] = y;
  dist[2] = z;
}

void vvVolDesc::setDist(const vvVector3& d)
{
  dist = d;
}

//----------------------------------------------------------------------------
/** Resize volume so that the longest edge becomes the length of len.
  This operation does not change the aspect ratio.
*/
void vvVolDesc::resizeEdgeMax(float len)
{
  float maxLen;                                   // maximum edge length

  vvDebugMsg::msg(1, "vvVolDesc::resizeEdgeMax()");

  // Determine volume dimensions in world space:
  maxLen = (float)ts_max((float)vox[0] * dist[0],
    (float)vox[1] * dist[1],
    (float)vox[2] * dist[2]);
  _scale = len / maxLen;
}

//----------------------------------------------------------------------------
float vvVolDesc::getChannelValue(int frame, size_t x, size_t y, size_t z, size_t chan)
{
  uint8_t* data = getRaw(frame);
  float fval;
  size_t index;
  size_t bpv = getBPV();
  int ival;

  index = bpv * (x + y * vox[0] + z * vox[0] * vox[1]) + chan * bpc;
  switch(bpc)
  {
    case 1: fval = float(data[index]); break;
    case 2: ival = data[index] * 256 + data[index+1]; fval = float(ival); break;
    case 4: fval = *((float*)(data + index)); break;
    default: assert(0); fval = 0.0f; break;
  }
  return fval;
}

//----------------------------------------------------------------------------
/** Gets data along a line in a 3D volume dataset using Bresenham's algorithm.
    Both line end points must lie within the volume. The Coordinate system is:
    <PRE>
           y
           |__ x
          /
         z
    </PRE>
    The volume data is arranged like this:
    <UL>
      <LI>origin is top left front
<LI>width in positive x direction
<LI>height in negative y direction
<LI>slices in negative z direction
</UL>
@param x0,y0,z0  line starting point in voxels
@param x1,y1,z1  line end point in voxels
@param resArray  array to store results
*/
void vvVolDesc::getLineHistData(int x0, int y0, int z0, int x1, int y1, int z1,
std::vector< std::vector< float > >& resArray)
{
  int xd, yd, zd;
  int x, y, z;
  int ax, ay, az;
  int sx, sy, sz;
  int dx, dy, dz;
  int index;

  uint8_t* data = getRaw();                         //raw.getData();
  size_t bpv = getBPV();

  x0 = ts_clamp(x0, 0, (int)vox[0]-1);
  x1 = ts_clamp(x1, 0, (int)vox[0]-1);
  y0 = ts_clamp(y0, 0, (int)vox[1]-1);
  y1 = ts_clamp(y1, 0, (int)vox[1]-1);
  z0 = ts_clamp(z0, 0, (int)vox[2]-1);
  z1 = ts_clamp(z1, 0, (int)vox[2]-1);

  dx = x1 - x0;
  dy = y1 - y0;
  dz = z1 - z0;

  ax = ts_abs(dx) << 1;
  ay = ts_abs(dy) << 1;
  az = ts_abs(dz) << 1;

  sx = ts_zsgn(dx);
  sy = ts_zsgn(dy);
  sz = ts_zsgn(dz);

  x = x0;
  y = y0;
  z = z0;

  resArray.clear();

  if (ax >= ts_max(ay, az))                       // x is dominant
  {
    yd = ay - (ax >> 1);
    zd = az - (ax >> 1);
    for (;;)
    {
      index = (z * vox[0] * vox[1] + y * vox[0] + x) * bpv;

      std::vector< float > tmp(chan);

      for (size_t i = 0; i < chan; i++)
      {
        switch (bpc)
        {
          case 1:
            tmp[i] = float(data[index + i]);
            break;
          case 2:
            tmp[i] = float(int(data[index + 2 * i] << 8) | int(data[index + 2 * i + 1]));
            break;
          case 4:
            tmp[i] = *((float*)(data + index + 4 * i));
            break;
          default:
            assert(0);
            break;
        }
      }

      resArray.push_back(tmp);

      // compute next voxel
      if (x == x1) return;
      if (yd >= 0)
      {
        y += sy;
        yd -= ax;
      }
      if (zd >= 0)
      {
        z += sz;
        zd -= ax;
      }
      x += sx;
      yd += ay;
      zd += az;
    }
  }
  else if (ay >= ts_max(ax, az))                  // y is dominant
  {
    xd = ax - (ay >> 1);
    zd = az - (ay >> 1);
    for (;;)
    {
      index = (z * vox[0] * vox[1] + y * vox[0] + x) * bpv;

      std::vector< float > tmp(chan);

      for (size_t i = 0; i < chan; i++)
      {
        switch (bpc)
        {
          case 1:
            tmp[i] = float(data[index + i]);
            break;
          case 2:
            tmp[i] = float(int(data[index + 2 * i] << 8) | int(data[index + 2 * i + 1]));
            break;
          case 4:
            tmp[i] = *((float*)(data + index + 4 * i));
            break;
          default:
            assert(0);
            break;
        }
      }

      resArray.push_back(tmp);

      // compute next voxel;
      if (y == y1) return;
      if (xd >= 0)
      {
        x += sx;
        xd -= ay;
      }
      if (zd >= 0)
      {
        z += sz;
        zd -= ay;
      }
      y += sy;
      xd += ax;
      zd += az;

    }
  }
  else if (az >= ts_max(ax, ay))                  // z is dominant
  {
    xd = ax - (az >> 1);
    yd = ay - (az >> 1);
    for (;;)
    {
      index = (z * vox[0] * vox[1] + y * vox[0] + x) * bpv;

      std::vector< float > tmp(chan);

      for (size_t i = 0; i < chan; i++)
      {
        switch (bpc)
        {
          case 1:
            tmp[i] = float(data[index + i]);
            break;
          case 2:
            tmp[i] = float(int(data[index + 2 * i] << 8) | int(data[index + 2 * i + 1]));
            break;
          case 4:
            tmp[i] = *((float*)(data + index + 4 * i));
            break;
          default:
            assert(0);
            break;
        }
      }

      resArray.push_back(tmp);

      // compute next voxel
      if (z == z1) return;
      if (xd >= 0)
      {
        x += sx;
        xd -= az;
      }
      if (yd >= 0)
      {
        y += sy;
        yd -= az;
      }
      z += sz;
      xd += ax;
      yd += ay;
    }
  }
}

//----------------------------------------------------------------------------
/** Find min and max data values and set real[0/1] accordingly.
 */
void vvVolDesc::setDefaultRealMinMax()
{
  float fMin, fMax;
  findMinMax(0, fMin, fMax);
  real[0] = fMin;
  real[1] = fMax;
}

//----------------------------------------------------------------------------
/** Create a height field from a single slice.
  The slice must have one data channel and can be any bpv.
  Voxels above the height field are set to zero (or real minimum if float).
  Voxels below the height field are set according to 'mode':
  0: zero (or real minimum if float)
  1: same value as height pixel
  @param slices number of slices to map heights to
  @param mode computation mode: 0=height surface, one voxel thick; 1=solid
  @param verbose true=show progress
  @return true if successful, false on error
*/
bool vvVolDesc::makeHeightField(size_t slices, int mode, bool verbose)
{
  float height = 0.f;                             // current height on scale 0..1

  uint8_t* rd;                                    // raw data of current source frame
  vvVector3 v;                                    // currently processed voxel coordinates [voxel space]
  uint8_t* newRaw;                                // raw data of current destination frame
  uint8_t *src, *dst;                             // source and destination volume data
  size_t newFrameSize;                            // new volume's frame size [voxels]
  ssize_t zPos;

  vvDebugMsg::msg(2, "vvVolDesc::makeHeightField()");

  if (vox[2] != 1)
  {
    cerr << "Height field conversion requires single slice volume." << endl;
    return false;
  }
  if (chan != 1)
  {
    cerr << "Height field conversion works only for single channel volumes." << endl;
    return false;
  }

  newFrameSize = vox[0] * vox[1] * slices * bpc;
  if (verbose) vvToolshed::initProgress(slices * frames);
  raw.first();
  for (size_t f=0; f<frames; ++f)
  {
    rd = raw.getData();
    newRaw = new uint8_t[newFrameSize];
    dst = newRaw;

    // Traverse destination data:
    for (ssize_t z=0; z<static_cast<ssize_t>(slices); ++z)
    {
      src = rd;
      for (ssize_t y=0; y<vox[1]; ++y)
      {
        for (ssize_t x=0; x<vox[0]; ++x)
        {
          // Calculate normalized height [0..1]:
          switch(bpc)
          {
            case 1: height = float(*src) / 255.0f; break;
            case 2: height = (float(*src) * 256.0f + float(*(src+1))) / 65535.0f; break;
            case 4: height = (*((float*)src) - real[0]) / (real[1] - real[0]); break;
            default: assert(0); break;
          }

          // Calculate voxel value:
          zPos = size_t(float(slices) * height);
          if (z > zPos)
          {
            memset(dst, 0, bpc);                  // above surface
          }
                                                  // on or below surface
          else if (z == zPos || mode==1) memcpy(dst, src, bpc);
          else memset(dst, 0, bpc);

          dst += bpc;
          src += bpc;
        }
      }
      if (verbose) vvToolshed::printProgress(z + slices * f);
    }
    raw.remove();
    if (f==0) raw.insertBefore(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    else raw.insertAfter(newRaw, vvSLNode<uint8_t*>::ARRAY_DELETE);
    raw.next();
  }
  vox[2] = slices;
  return true;
}

//----------------------------------------------------------------------------
/** This function adds one or three data channels to the volume containing
  the gradient magnitues or vector gradients for one of the other channels.
  No gradients are calculated for edge voxels because values outside of the
  volume are undefined.
  @param srcChan channel to calculate gradient magnitues for [0..numChan-1]
  @param gradType type of gradient: magnitude (adds 1 channel),
                  or gradient vectors (adds 3 channels)
*/
void vvVolDesc::addGradient(size_t srcChan, GradientType gradType)
{
  const float SQRT3 = float(sqrt(3.0));
  const char* GRADIENT_MAGNITUDE_CHANNEL_NAME = "GRADMAG";
  const char* GRADIENT_X_CHANNEL_NAME = "GRADIENT_X";
  const char* GRADIENT_Y_CHANNEL_NAME = "GRADIENT_Y";
  const char* GRADIENT_Z_CHANNEL_NAME = "GRADIENT_Z";
  uint8_t* src;                                   // pointer to current source channel
  uint8_t* dst;                                   // pointer to current destination channel (first of three if gradient vectors)
  uint8_t* surrPtr[6];                            // pointers to surrounding voxels: next and previous values along coordinate axes
  float surr[6];                                  // values of surrounding voxels
  float diff[3];                                  // differences along coordinate axes
  float grad;                                     // gradient value as floating point
  int iGrad;                                      // gradient value as integer
  size_t sliceVoxels;                             // number of voxels per slice
  size_t sliceBytes;                              // number of bytes per slice
  size_t lineBytes;                               // number of bytes per volume line
  size_t bpv;                                     // bytes per voxel
  size_t offset;                                  // offset from source channel to destination channel

  size_t numNewChannels;

  // Add new channels and name them:
  numNewChannels = (gradType==GRADIENT_MAGNITUDE) ? 1 : 3;
  convertChannels(chan + numNewChannels);
  if (gradType==GRADIENT_MAGNITUDE)
  {
    setChannelName(chan-1, GRADIENT_MAGNITUDE_CHANNEL_NAME);
  }
  else
  {
    setChannelName(chan-3, GRADIENT_X_CHANNEL_NAME);
    setChannelName(chan-2, GRADIENT_Y_CHANNEL_NAME);
    setChannelName(chan-1, GRADIENT_Z_CHANNEL_NAME);
  }

  sliceVoxels = getSliceVoxels();
  sliceBytes = getSliceBytes();
  bpv = bpc * chan;
  lineBytes = bpv * vox[0];
  offset = bpc * (chan - srcChan - numNewChannels);

  // Add gradient magnitudes to every frame:
  for (size_t f=0; f<frames; ++f)
  {
    // Calculate gradient magnitudes for non-edge voxels:
    src = getRaw(f) + bpv * (1 + vox[0] + sliceVoxels) + bpc * srcChan;
    for(ssize_t z=1; z<vox[2]-1; ++z)
    {
      for(ssize_t y=1; y<vox[1]-1; ++y)
      {
        for(ssize_t x=1; x<vox[0]-1; ++x)
        {
          // Calculate gradient from surrounding voxels:
          surrPtr[0] = src - bpv;
          surrPtr[1] = src + bpv;
          surrPtr[2] = src - lineBytes;
          surrPtr[3] = src + lineBytes;
          surrPtr[4] = src - sliceBytes;
          surrPtr[5] = src + sliceBytes;
          for (int i=0; i<6; ++i)
          {
            switch(bpc)
            {
              case 1: surr[i] = float(*surrPtr[i]) / 255.0f; break;
              case 2: surr[i] = float((int(*surrPtr[i]) << 8) | int(*(surrPtr[i]+1))) / 65535.0f; break;
              case 4: surr[i] = *((float*)surrPtr[i]); break;
              default: surr[i] = 0.0f; assert(0); break;
            }
          }

          diff[0] = surr[1] - surr[0];
          diff[1] = surr[3] - surr[2];
          diff[2] = surr[5] - surr[4];

          // Store gradient in new channels:
          dst = src + offset;
          if (gradType==GRADIENT_MAGNITUDE)
          {
                                                  // reduce value to range 0..1
            grad = float(sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])) / SQRT3;
            grad = ts_clamp(grad, 0.0f, 1.0f);
            switch(bpc)
            {
              case 1: *dst = int(grad * 255.0f); break;
              case 2: iGrad = int(grad * 65535.0f);
              *dst = iGrad >> 8; *(dst+1) = iGrad & 0xff; break;
              case 4: *((float*)dst) = grad; break;
              default: assert(0); break;
            }
          }
          else
          {
            for (int i=0; i<3; ++i)
            {
              switch(bpc)
              {
                case 1:
                  iGrad = int((diff[i] + 1.0f) * 127.5f);
                  *(dst+i) = ts_clamp(iGrad, 0, 255);
                  break;
                case 2:
                  iGrad = int((diff[i] + 1.0f) * 32767.5f);
                  *(dst+2*i) = iGrad >> 8;
                  *(dst+2*i+1) = iGrad & 0xff;
                  break;
                case 4:
                  *((float*)(dst+i*4)) = diff[i];
                  break;
                default: assert(0); break;
              }
            }
          }
          src += bpv;
        }
        src += 2 * bpv;
      }
      src += 2 * lineBytes;
    }
  }
}

//----------------------------------------------------------------------------
/** Calculate mean and variance for the 3x3x3 neighborhood of a voxel.
*/
void vvVolDesc::voxelStatistics(size_t frame, size_t c, ssize_t x, ssize_t y, ssize_t z, float& mean, float& variance)
{
  uint8_t* raw;
  double sumSquares = 0.0;
  double sum = 0.0;
  float diff;
  float scalar = 0.f;
  size_t offset;
  ptrdiff_t dx,dy,dz;     // offsets to neighboring voxels
  size_t bpv;
  int mode;   // 0=mean, 1=variance
  size_t i;
  size_t numSummed;

  raw = getRaw(frame);
  bpv = bpc * chan;
  offset = bpv * (x + y * vox[0] + z * vox[0] * vox[1]) + bpc * c;
  for (mode=0; mode<2; ++mode)
  {
    numSummed = 0;
    for (dx=-1; dx<=1; ++dx)
    {
      if (x+dx > vox[0]-1) continue;
      for (dy=-1; dy<=1; ++dy)
      {
        if (y+dy > vox[1]-1) continue;
        for (dz=-1; dz<=1; ++dz)
        {
          if (z+dz > vox[2]-1) continue;
          i = offset + bpv * (dx + dy * vox[0] + dz * vox[0] * vox[1]);
          switch (bpc)
          {
            case 1:
              scalar = float(raw[i]);
              break;
            case 2:
              scalar = float((int(raw[i]) << 8) | int(raw[i+1]));
              break;
            case 4:
              scalar = *((float*)(raw+i));
              break;
            default: assert(0); break;
          }
          if (mode==0)
          {
            sum += scalar;
            ++numSummed;
          }
          else
          {
            diff = scalar - mean;
            sumSquares += diff * diff;
            ++numSummed;
          }
        }
      }
    }
    if (mode==0) mean = float(sum / double(numSummed));
    else variance = float(sumSquares / double(numSummed));
  }
}

//----------------------------------------------------------------------------
/** This function adds a data channels to the volume containing
  the variance in the 3x3x3 voxel neighborhood for one of the other channels.
  @param srcChan channel to calculate variance for [0..numChan-1]
*/
void vvVolDesc::addVariance(size_t srcChan)
{
  const char* VARIANCE_CHANNEL_NAME = "VARIANCE";
  uint8_t* src;                                   // pointer to current source channel
  uint8_t* dst;                                   // pointer to current destination channel (first of three if gradient vectors)
  float mean;                                     // mean data value in a frame
  float variance;
  int iVar;                                       // variance as integer
  size_t bpv;                                     // bytes per voxel
  size_t voxelOffset;                             // offset from source channel to destination channel [bytes]

  // Add new channel and name it:
  convertChannels(chan + 1);
  setChannelName(chan-1, VARIANCE_CHANNEL_NAME);

  bpv = bpc * chan;
  voxelOffset = bpc * (chan - srcChan - 1);

  // Add variance to every frame:
  for (size_t f=0; f<frames; ++f)
  {
    // Calculate variance for all voxels, including edge voxels:
    src = getRaw(f) + bpc * srcChan;
    for(ssize_t z=0; z<vox[2]; ++z)
    {
      for(ssize_t y=0; y<vox[1]; ++y)
      {
        for(ssize_t x=0; x<vox[0]; ++x)
        {
          voxelStatistics(f, srcChan, x, y, z, mean, variance);
          dst = src + voxelOffset;
          switch(bpc)
          {
            case 1: *dst = int(variance * 255.0f);
                    break;
            case 2: iVar = int(variance * 65535.0f);
                    *dst = iVar >> 8; *(dst+1) = iVar & 0xff;
                    break;
            case 4: *((float*)dst) = variance;
                    break;
            default: assert(0); break;
          }
          src += bpv;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
/** Update bin limits array for HDR transfer functions.
  @param numValues if >0 then use subset of volume for sampling
  @param skipWidgets if true, algorithm ignores data values under Skip widgets
  @param cullDup remove duplicate values from value array to avoid cluttering bins with same value
  @param lockRange if true, realMin/realMax won't be modified
  @param binning binning type
  @param transOp true=transfer opacities to bin space
*/
void vvVolDesc::updateHDRBins(size_t numValues, bool skipWidgets, bool cullDup, bool lockRange, BinningType binning, bool transOp)
{
  vvTFSkip* sw;
  uint8_t* srcData;
  float* sortedData;
  float* sortedTmp;
  float* opacities=NULL;
  double sumOpacities=0.;    // sum of all opacities
  double opacityPerBin;
  double localSum;
  float min,max;
  float valuesPerBin;
  size_t numVoxels;
  size_t index, minIndex, maxIndex, numSkip;
  size_t before;

  assert(binning!=LINEAR);    // this routine supports only iso-data and opacity-weighted binning
  if (bpc!=4 || chan!=1)
  {
    cerr << "updateHDRBins() works only on single channel float data" << endl;
    return;
  }

  _transOp = transOp;

  vvStopwatch stop;
  stop.start();
  cerr << endl << "Starting HDR timer" << endl;
  cerr << "Creating HDR data array...";
  assert(_hdrBinLimits);
  srcData = getRaw();
  numVoxels = getFrameVoxels();
  if (numValues>0) numVoxels = ts_min(numVoxels, numValues);
  sortedData = new float[numVoxels+2];    // +2 for min/max of data range
  if (numVoxels == getFrameVoxels())  // can the entire data array be sorted?
  {
    memcpy(sortedData, srcData, getFrameBytes());
  }
  else  // create monte carlo volume
  {
    for (size_t i=0; i<numVoxels; ++i)
    {
      index = size_t(numVoxels * float(rand()) / float(RAND_MAX));
      sortedData[i] = *((float*)(srcData + (sizeof(float) * index)));
    }
  }
  cerr << stop.getDiff() << " sec" << endl;

  // Make sure min and max of data range are included in data:
  if (lockRange)
  {
    sortedData[numVoxels]   = real[0];
    sortedData[numVoxels+1] = real[1];
    numVoxels += 2;
  }

  // Sort data:
  cerr << "Sorting data array...";
  sort(sortedData, sortedData+numVoxels);   // requires #include <algorithm>
  cerr << stop.getDiff() << " sec" << endl;

  // Trim beginning and end of sorted data array to remove values below/above realMin/realMax:
  if (lockRange)
  {
    cerr << "Trimming values to maintain range...";

    // Trim values below realMin:
    size_t i;
    for(i=0; i<numVoxels && sortedData[i] < real[0]; ++i)
    {
      minIndex = i;  // find index of realMin
    }
    minIndex = ts_clamp(i, size_t(0), numVoxels-1);
    if (minIndex > 0)
    {
      memcpy(&sortedData[0], &sortedData[minIndex], sizeof(float) * (numVoxels - minIndex));
      numVoxels -= minIndex;
    }

    // Trim values above realMax:
    // find index of realMax
    for(i=numVoxels-1; i>0 && sortedData[i] > real[1]; --i)
       ;
    maxIndex = ts_clamp(i, size_t(0), numVoxels-1);
    numVoxels -= (numVoxels-1-maxIndex);
    cerr << stop.getDiff() << " sec" << endl;
  }

  // Remove areas covered by TFSkip widgets:
  if (skipWidgets)
  {
    cerr << "Removing skipped regions from data array...";
    before = numVoxels;
    numSkip = 0;
    for (std::vector<vvTFWidget*>::const_iterator it = tf._widgets.begin();
         it != tf._widgets.end(); ++it)
    {
      if ((sw=dynamic_cast<vvTFSkip*>(*it))!=NULL)
      {
        min = sw->_pos[0] - sw->_size[0] / 2.0f;
        max = sw->_pos[0] + sw->_size[0] / 2.0f;

        // Find index of beginning of skip area:
        size_t i;
        for(i=0; i<numVoxels && sortedData[i] < min; ++i)
           ;
        if (i<numVoxels)   // is skip region outside of data array?
        {
          minIndex = i;

          // Find index of end of skip area:
          for(i=minIndex; i<numVoxels && sortedData[i] < max; ++i)
             ;
          maxIndex = ts_clamp(i, size_t(0), numVoxels-1);

          // Cut values:
          numSkip = maxIndex - minIndex + 1;
          if (maxIndex<numVoxels-1)
          {
            memcpy(&sortedData[minIndex], &sortedData[maxIndex+1], sizeof(float) * (numVoxels - maxIndex - 1));
          }
          numVoxels -= numSkip;
        }
      }
    }
    cerr << stop.getDiff() << " sec" << endl;
    cerr << (before - numVoxels) << " voxels removed (" << (100.0f * float(numSkip) / float(getFrameVoxels())) << "%)" << endl;
  }

  // Remove duplicate values from array:
  if (cullDup)
  {
    before = numVoxels;
    cerr << "Removing duplicate values...";
    sortedTmp = new float[numVoxels];
    sortedTmp[0] = sortedData[0];
    size_t j=0;
    for (size_t i=1; i<numVoxels; ++i)
    {
      if (sortedData[i] != sortedTmp[j])
      {
        ++j;
        sortedTmp[j] = sortedData[i];
      }
    }
    numVoxels = j+1;
    memcpy(sortedData, sortedTmp, numVoxels * sizeof(float));
    delete[] sortedTmp;
    cerr << stop.getDiff() << " sec" << endl;
    cerr << (before - numVoxels) << " voxels removed" << endl;
  }

  // Create array with opacity values for the data values:
  if (binning==OPACITY)
  {
    cerr << "Creating opacity array...";
    sumOpacities = 0.0;
    opacities = new float[numVoxels * sizeof(float)];
    for (size_t i=0; i<numVoxels; ++i)
    {
      opacities[i] = tf.computeOpacity(sortedData[i]);
      sumOpacities += opacities[i];
    }
    cerr << stop.getDiff() << " sec" << endl;
  }

  // Determine bin limits:
  cerr << "Determining bin limits...";
  switch (binning)
  {
    case ISO_DATA:
      valuesPerBin = float(numVoxels) / float(NUM_HDR_BINS);
      for (size_t i=0; i<NUM_HDR_BINS; ++i)
      {
        index = size_t(float(i) * valuesPerBin);
        index = ts_clamp(index, size_t(0), numVoxels-1);
        _hdrBinLimits[i] = sortedData[index];
      }
      break;
    case OPACITY:
    {
      opacityPerBin = sumOpacities / NUM_HDR_BINS;
      localSum = 0.0;
      size_t j = 0;    // index to current voxel
      for (size_t i=0; i<NUM_HDR_BINS; ++i)
      {
        while (localSum<opacityPerBin)
        {
          if (j<numVoxels)
          {
            localSum += opacities[j];
            ++j;
            if (localSum>=opacityPerBin)
            {
              _hdrBinLimits[i] = sortedData[j-1];
              localSum = 0.0;
              break;
            }
          }
          else
          {
            _hdrBinLimits[i] = sortedData[numVoxels-1];
            localSum = 0.0;
            break;
          }
        }
      }
      break;
    }
    case LINEAR:
      // cannot happen
      break;
  }
  cerr << stop.getDiff() << " sec" << endl;

  // Do first and last data entries differ?
  if (sortedData[0] == sortedData[numVoxels-1])
  {
    cerr << "volume too sparse for HDR monte carlo sampling" << endl;
  }
  else    // adjust real min/max to min/max found in volume, unless not desired by the user
  {
    if (!lockRange)
    {
      real[0] = sortedData[0];
      real[1] = sortedData[numVoxels-1];
    }
  }

  delete[] opacities;
  delete[] sortedData;

  cerr << "Total HDR execution time: " << stop.getTime() << " sec" << endl << endl;
}

//----------------------------------------------------------------------------
/** Map floating point data value to integer. Uses high dynamic range (HDR)
  techniques if _binOpacityWeighted or _binIsoData are true, otherwise
  it clamps the value between real[0] and real[1] and linearly maps the value
  to an 8bit integer.
  @param fval floating point data value
  @return 8bit integer value [0..255]
*/
int vvVolDesc::mapFloat2Int(float fval)
{
  int ival;

  switch(_binning)
  {
    case LINEAR:
      fval = ts_clamp(fval, real[0], real[1]);
      return int((fval - real[0]) / (real[1] - real[0]) * 255.0f);
    case ISO_DATA:
    case OPACITY:
      ival = findHDRBin(fval);
      return ival;
    default: assert(0);
      return -1;
  }
}

//----------------------------------------------------------------------------
/** Returns the number of the bin the floating point value is in.
*/
int vvVolDesc::findHDRBin(float fval)
{
  for (size_t i=0; i<NUM_HDR_BINS; ++i)
  {
    if (_hdrBinLimits[i] > fval)
    {
      return ts_max(i-1, size_t(0));
    }
  }
  return NUM_HDR_BINS-1;
}

//----------------------------------------------------------------------------
/** Create 1D texture with bin limits.
  @param texture pointer to _allocated_ space for width * 4 bytes
  @param width texture width [pixels]
*/
void vvVolDesc::makeBinTexture(uint8_t* texture, size_t width)
{
  float range;

  memset(texture, 0, width * 4);    // initialize with transparent texels
  if (bpc==4)
  {
    range = real[1] - real[0];
    for (size_t i=0; i<NUM_HDR_BINS; ++i)
    {
      size_t index = size_t((_hdrBinLimits[i] - real[0]) / range * float(width-1));
      index = ts_clamp(index, size_t(0), size_t(width-1));
      texture[4*index]   = 0;
      texture[4*index+1] = 0;
      texture[4*index+2] = 0;
      texture[4*index+3] = 255;   // make tick marks opaque
    }
  }
}

static size_t index(size_t i, size_t j, size_t k, size_t d0, size_t d1, size_t d2)
{
  (void)d2;
  return i+j*d0+k*d0*d1;
}

//----------------------------------------------------------------------------
/** Create 2 arrays containing min and max for downsample^3 sized blocks
 * memory has to be allocated by the user
  @param minArray minimum data
  @param maxArray maximum data
  @param downsample edge length of block for which data is summarized
  @param channel no. of channel to extract data from
  @param frame no. of channel to extract data from
*/
void vvVolDesc::computeMinMaxArrays(uchar *minArray, uint8_t *maxArray, ssize_t downsample, size_t channel, int frame) const
{
  vvDebugMsg::msg(2, "vvVolDesc::computeMinMaxArrays()");

  if(bpc != 1)
    return;

  ssize_t v0 = (vox[0]+downsample-1)/downsample;
  ssize_t v1 = (vox[1]+downsample-1)/downsample;
  ssize_t v2 = (vox[2]+downsample-1)/downsample;

  uint8_t *raw = const_cast<vvVolDesc *>(this)->getRaw(frame);
  for(ssize_t k=0; k<vox[2]; k += downsample)
  {
    for(ssize_t j=0; j<vox[1]; j += downsample)
    {
      for(ssize_t i=0; i<vox[0]; i += downsample)
      {
        uint8_t cMin = 255, cMax = 0;
        {
          for(ssize_t kk=k; kk<vox[2] && kk<k+downsample; ++kk)
            for(ssize_t jj=j; jj<vox[1] && jj<j+downsample; ++jj)
            {
              for(ssize_t ii=i; ii<vox[0] && ii<i+downsample; ++ii)
              {
                uint8_t v = raw[chan*index(ii, jj, kk, vox[0], vox[1], vox[2])+channel];
                if(cMin > v)
                  cMin = v;
                if(cMax < v)
                  cMax = v;
              }
            }
        }
        size_t idx = index(i/downsample, j/downsample, k/downsample, v0, v1, v2);
        minArray[idx] = cMin;
        maxArray[idx] = cMax;
      }
    }
  }
}

vvssize3 vvVolDesc::voxelCoords(const vvVector3& objCoords) const
{
  vvVector3 fltVox2 = vvVector3(static_cast<float>(vox[0]) * 0.5f,
                                static_cast<float>(vox[1]) * 0.5f,
                                static_cast<float>(vox[2]) * 0.5f);
  vvVector3 obj = objCoords;
  for (size_t i = 0; i < 3; ++i)
  {
    obj[i] /= dist[i];
    obj[i] /= _scale;
  }

  return vvssize3(static_cast<ssize_t>(obj[0] + fltVox2[0]),
                  static_cast<ssize_t>(obj[1] + fltVox2[1]),
                  static_cast<ssize_t>(obj[2] + fltVox2[2]));
}

vvVector3 vvVolDesc::objectCoords(const vvssize3& voxCoords) const
{
  vvVector3 fltVox2 = vvVector3(static_cast<float>(vox[0]) * 0.5f,
                                static_cast<float>(vox[1]) * 0.5f,
                                static_cast<float>(vox[2]) * 0.5f);
  vvVector3 result =  vvVector3(static_cast<float>(voxCoords[0]) - fltVox2[0],
                                static_cast<float>(voxCoords[1]) - fltVox2[1],
                                static_cast<float>(voxCoords[2]) - fltVox2[2]);
  for (size_t i = 0; i < 3; ++i)
  {
    result[i] *= dist[i];
    result[i] *= _scale;
  }
  return result;
}

///// EOF /////
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
