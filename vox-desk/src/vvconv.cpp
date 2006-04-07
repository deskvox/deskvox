// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
// 
// This file is part of DeskVOX.
//
// DeskVOX is free software; you can redistribute it and/or
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "vvvirvo.h"
#include "vvconv.h"
#include "vvfileio.h"
#include "vvdebugmsg.h"
#include "vvtokenizer.h"
#include <sstream>

using namespace std;

//----------------------------------------------------------------------------
/// Constructor
vvConv::vvConv()
{
  int i;

  vd = new vvVolDesc();
  srcFile       = NULL;
  dstFile       = NULL;
  files         = 1;
  increment     = 1;
  showHelp      = false;
  fileInfo      = false;
  overwrite     = false;
  bpchan        = -1;
  sphere        = false;
  outer         = 0;
  inner         = 0;
  heightField   = false;
  hfHeight      = 0;
  hfMode        = 0;
  crop          = false;
  croptime      = false;
  drawLine      = false;
  drawBox       = false;
  lineColor     = 0;
  boxColor      = 0;
  resize        = false;
  ipt           = vvVolDesc::NEAREST;
  resizeFactor  = 0.0f;
  flip          = false;
  flipAxis      = vvVolDesc::X_AXIS;
  rotate        = false;
  rotAxis       = vvVolDesc::X_AXIS;
  rotDir        = 0;
  setDist       = false;
  setRange      = false;
  swap          = false;
  sign          = false;
  shift         = false;
  bitshift      = false;
  bshiftDist    = 0;
  importTF      = false;
  removeTF      = false;
  importFile    = NULL;
  loadRaw       = false;
  rawBPC        = 1;
  rawCh         = 1;
  rawWidth = rawHeight = rawSlices = 0;
  rawSkip       = 0;
  newRange[0]   = 0.0f;
  newRange[1]   = 1.0f;
  statistics    = false;
  fillRange     = false;
  dicomRename   = false;
  leicaRename   = false;
  compression   = true;
  animTime      = 0.0f;
  deinterlace   = false;
  zoomData      = false;
  loadXB7       = false;
  xb7Size       = 0;
  xb7Param      = 0;
  xb7Global     = false;
  loadCPT       = false;
  cptSize       = 0;
  cptParam      = 0;
  cptGlobal     = false;
  setPos        = false;
  histogram     = false;
  histType      = 0;
  blend         = false;
  blendFile     = NULL;
  blendType     = 0;
  channels      = -1;
  setIcon       = false;
  iconFile      = NULL;
  makeIcon      = false;
  makeVolume    = -1;
  makeIconSize  = 0;
  getIcon       = false;
	swapChannels  = false;
  extractChannel = false;
  addChannel    = false;
  addFile       = NULL;
  autoRealRange = false;
  invertVoxelOrder = false;
  lineAverage = 0;
  sections = 0;
  pinhole = 0;
  height = 0;
  width = 0;
  time = 0;
  thickness = 0;
  lasercount = 0;
  starttime = 999999;
  endtime = 0;
  mergeType     = 0;

  for (i=0; i<3; ++i)
  {
    chan[i] = -1;
    cropPos[i]   = 0;
    cropSize[i]  = 0;
    newSize[i]   = 0;
    newDist[i]   = 1.0f;
    shiftDist[i] = 0;
    lineStart[i] = 0;
    lineEnd[i]   = 0;
    boxStart[i]  = 0;
    boxEnd[i]    = 0;
    posObj[i]    = 0.0f;
    zoomRange[i] = 0;
    extract[i]   = 0.0f;
  }
  for (i=0; i<2; ++i)
  {
    cropSteps[i]  = 0;
		swapChan[i]   = 0;
  } 
}

//----------------------------------------------------------------------------
/// Destructor
vvConv::~vvConv()
{
  delete vd;
}

//----------------------------------------------------------------------------
/** Reads one or more volume files, processes conversion options, and
 combines data.
 @return true if ok, false on error
*/
bool vvConv::readVolumeData()
{
  vvFileIO*  fio;
  vvVolDesc* newVD;
  char* filename;             // currently processed file name
  int   file  = 0;            // index of current file
  bool  done = false;
  vvFileIO::ErrorType error;
  int i;

  vvDebugMsg::msg(1, "vvConv::readVolumeData()");

  filename = new char[strlen(srcFile) + 1];
  strcpy(filename, srcFile);
  fio = new vvFileIO();

  if (leicaRename)
  {
    vd = new vvVolDesc(filename);

    string tempname = srcFile;
    int start = tempname.find("_Series",0);

    if (start == -1)
      start = tempname.find("_Image", 0);

    char* seriesname =(char*)((tempname.substr(0, start)).c_str());
    strcat(seriesname, ".txt");
    
    cerr << "Merging Leica Files" << endl;
    fio->mergeFiles(vd, files, increment, vvVolDesc::VV_MERGE_SLABS2VOL);
    cerr << "Done merging Leica files" << endl;


    
    FILE *fp;
    if ((fp = fopen(seriesname, "rb")) == NULL)
       cerr << "Error: Cannot open " << seriesname << " file.";
    else
    {
      // File found.  Now to determine values
      cerr << "Opened: " << seriesname << endl;

      vvTokenizer* tok = new vvTokenizer(fp);
      tok->setParseNumbers(true);
      vvTokenizer::TokenType ttype;
      
      laser = new int[vd->chan];
      intensity = new int[vd->chan];
      gain = new int[vd->chan];
      offset = new int[vd->chan];

      for (int i = 0; i < vd->chan; i++)
      {
        laser[i] = 0;
        intensity[i] = 0;
        gain[i] = 0;
        offset[i] = 0;
      }
      
      bool done = false;
      while(!done) {
        ttype = tok->nextToken();
        if (strcmp(tok->sval, "SCANNER") == 0)
        {
          ttype = tok->nextToken();                  
          if (strcmp(tok->sval, "INFORMATION") ==0)
          {
            ttype = tok->nextToken();
            if (strcmp(tok->sval, "#0") == 0)
             done = true;
          }
        }
      }
      

      done = false;
      while(!done)
      {    
        tok->nextToken();

        if (strcmp(tok->sval, "Line-Average") ==0)
        {
          tok->nextToken();
          lineAverage = int(tok->nval);
        }         
        else if (strcmp(tok->sval, "Sections") == 0)
        {
          tok->nextToken();
          sections = int(tok->nval);
        }
        else if (strcmp(tok->sval, "Pinhole") == 0)
        {
          tok->nextToken();
          if (strcmp(tok->sval, "[airy]") == 0)
          {
            tok->nextToken();
            pinhole = int(tok->nval);
          }
        }
        else if (strcmp(tok->sval, "Format-Width") == 0)
        {
          tok->nextToken();
          width = int(tok->nval);
        }              
        else if (strcmp(tok->sval, "Format-Height") == 0)
        {
          tok->nextToken();
          height = int(tok->nval);
        }              
       else if (strcmp(tok->sval, "Voxel-Depth") ==0)
        {
          tok->nextToken();
          if (strcmp(tok->sval, "[")==0)
          {
            tok->nextToken();
            if (strcmp(tok->sval, "m]") == 0)
            {
              tok->nextToken();
              thickness = float(tok->nval);
            }
          }
        }
        else if (strcmp(tok->sval, "TIME") == 0)
        {
          tok->nextToken();
          if (strcmp(tok->sval, "INFORMATION") ==0)
            done = true;
        }
      }

      // Grab time information
      
      done = false;
      while(!done)
      {
        tok->setWhitespaceCharacter('_');
        tok->setWhitespaceCharacter(',');
        tok->nextToken();
        if (strcmp(tok->sval, "Stamp") == 0)
        {
          tok->nextToken();
          tok->nextToken();
          tok->nextToken();
          
          string stime = tok->sval;
          int hour = 0, minute = 0, second = 0;
          stringstream ss;
          ss << stime.substr(0,2);
          ss >> hour;

          ss.clear();
          ss << stime.substr(3,2);
          ss >> minute;

          ss.clear();
          ss << stime.substr(6,2);
          ss >> second;
          
          int time = 60 * 60 * hour + 60 * minute + second;

          if (time < starttime)
            starttime = time;
          if (time > endtime)
            endtime = time;
        }

        if (strcmp(tok->sval, "LUT") ==0)
          done = true;
      }
      tok->setAlphaCharacter(',');
     
      done = false;
      while (!done)
      {
        if (strcmp(tok->sval, "LUT") == 0)
        {
          tok->nextToken();
          if (strcmp(tok->sval, "DESCRIPTION") == 0)
          {
            tok->nextToken();
            if (strcmp(tok->sval, "#0") == 0)
            {
              bool bLUT = false;
              while(!bLUT)
              {
                tok->nextToken();
                if (strcmp(tok->sval, "LUT") == 0)
                {
                  tok->nextToken();

                  int ch = int(tok->nval);

                  tok->nextToken();
                  tok->nextToken();

                  if (strcmp(tok->sval, "Blue") == 0)
                    chan[2] = ch;
                  else if (strcmp(tok->sval, "Green") == 0)
                    chan[1] = ch;
                  else if (strcmp(tok->sval, "Red") == 0)
                    chan[0] = ch;
                }
                if (strcmp(tok->sval, "SEQUENTIAL") == 0)
                  bLUT = true;
              }
            }
          }
        }
        if (strcmp(tok->sval, "HARDWARE") == 0)
        {
          tok->nextToken();
          if (strcmp(tok->sval, "PARAMETER") == 0)
          {
            tok->nextToken(); 
            tok->nextToken();
              int map[4];            
              for (int i = 0; i < 4; i++)
                map[i] = 0;
              int curmap = 0;
              int assignedlaser = 0;
              int addinglaser = 0;
              bool bDone = false;
              while (!bDone)
              {
                tok->nextToken();
              
                if (strcmp(tok->sval, "AOTF") == 0)
                {
                  tok->nextToken();
                  string temp = tok->sval;
          
                  char* las = (char*)((temp.substr(1, temp.length() - 2)).c_str());

                  istringstream iss(las);
                  int curlaser;
                  iss >> curlaser;
  
                  tok->nextToken();

                  if (tok->nval != 0)
                  {
                    map[curmap] = curlaser;
                    curmap++;
                    bool addlaser = true;
                    addinglaser++;
                    for (int i = 0; i < lasercount; i++)
                      if (laser[i] == curlaser)
                        addlaser = false;
                    if (addlaser)
                    {
                      laser[lasercount] = curlaser;
                      intensity[lasercount] = int(tok->nval);
                      lasercount++;
                    }
                  }
                }
                
                
                if (strcmp(tok->sval, "PMT") == 0)
                {
                  tok->nextToken();
                  tok->nextToken();
                  if (strcmp(tok->sval, "Active") == 0)
                  {
                    tok->nextToken();
                    tok->nextToken();
                    tok->nextToken();
                    tok->nextToken();

                    int thelaser = map[assignedlaser];
                    int arraylaser = 0;
                    for (int i = 0; i < lasercount; i++)
                      if (laser[i] == thelaser)
                        arraylaser = i;
                    assert(arraylaser >= 0);
                    
                    if (addinglaser > 1)
                      tok->nextToken();
                    
                    if (strcmp(tok->sval, "(Offs.)") == 0)
                      tok->nextToken();

                    offset[arraylaser] = int(tok->nval);
                        
                    tok->nextToken();
                    tok->nextToken();
                    tok->nextToken();
                    tok->nextToken();

                    gain[arraylaser] = int(tok->nval);

                    assignedlaser++;
                  }
                }
                  
                    
                if (strcmp(tok->sval, "SCANNER") ==0)
                  bDone = true;
              }
            }
          
        }
        
        tok->setWhitespaceCharacter('_');
        tok->setWhitespaceCharacter(',');
        tok->nextToken();
        if (strcmp(tok->sval, "Stamp") == 0)
        {
          tok->nextToken();
          tok->nextToken();
          tok->nextToken();
          
          string stime = tok->sval;
          int hour = 0, minute = 0, second = 0;
          stringstream ss;
          ss << stime.substr(0,2);
          ss >> hour;

          ss.clear();
          ss << stime.substr(3,2);
          ss >> minute;

          ss.clear();
          ss << stime.substr(6,2);
          ss >> second;
          
          int time = 60 * 60 * hour + 60 * minute + second;

          if (time < starttime)
            starttime = time;
          if (time > endtime)
            endtime = time;
        }
        tok->setAlphaCharacter(',');


        if (tok->nextToken() == vvTokenizer::VV_EOF)
          done = true;
      }
      delete tok;

    }
    delete fp;
    
    if (vd->chan == 1)
    {
      vd->convertChannels(vd->chan + 1);
      vd->convertChannels(vd->chan + 1);
    }
    if (vd->chan == 2)
      vd->convertChannels(vd->chan + 1);

    int red = chan[0];
    int green = chan[1];
    int blue = chan[2];

    cerr << red << " " << green << " " << blue << endl;
    
    if (red == -1)
    {
      if (blue == 0)
        vd->swapChannels(0,2,false);
      if (green == 0)
        vd->swapChannels(0,1,false);
    }         
    else if (red != -1)
      vd->swapChannels(0,red, false);

    

    if (green == 0)
      green = red;
    if (blue == 0)
      blue = red;

    if (green != 1)
      vd->swapChannels(1,2, false);
    
    return true;
  }
  while (!done)
  {
    // Load current file:
    cerr << "Loading file " << (file+1) << ": " << filename << endl;
    newVD = new vvVolDesc(filename);
    if (loadRaw)
    {
      error = fio->loadRawFile(newVD, rawWidth, rawHeight, rawSlices, rawBPC, rawCh, rawSkip);
    }
    else if (loadXB7)
    {
      error = fio->loadXB7File(newVD, xb7Size, xb7Param, xb7Global);
    }
    else if (loadCPT)
    {
      error = fio->loadCPTFile(newVD, cptSize, cptParam, cptGlobal);
    }
    else if (makeVolume>-1)
    {
      newVD->computeVolume(makeVolume, makeVolumeSize[0], makeVolumeSize[1], makeVolumeSize[2]);
      error = vvFileIO::OK;
    }
    else error = fio->loadVolumeData(newVD);
    
    if (error != vvFileIO::OK)
    {
      cerr << "Cannot load file: " << filename << endl;
      delete fio;
      delete newVD;
      delete[] filename;
      return false;
    }

    newVD->printInfoLine("Loaded: ");

    // Make data modifications for each input file:
    modifyInputFile(newVD);

    // Merge new data to previous data:
    if (newVD->vox[2] == 1)
    {
      if (mergeType==2)
      {
        vd->merge(newVD, vvVolDesc::VV_MERGE_VOL2ANIM);
      }
      else vd->merge(newVD, vvVolDesc::VV_MERGE_SLABS2VOL);
    }
    else
    {
      if (mergeType==1) cerr << "Cannot merge volumes to another volume." << endl;
      else vd->merge(newVD, vvVolDesc::VV_MERGE_VOL2ANIM);
    }

    delete newVD;       // now the new VD can be released

    // Find the next file:
    ++file;
    if (file < files || files==0) 
    {
      for (i=0; i<increment && !done; ++i)
      {
        if (!vvToolshed::increaseFilename(filename))
        {
          cerr << "Cannot increase filename '" << filename << "'." << endl;
          done = true;
        }
      }

      if (!done && !vvToolshed::isFile(filename))
      {
        cerr << "File '" << filename << "' expected but not found." << endl;
        done = true;
      }
    }
    else done = true;
  }
  delete[] filename;
  delete fio;

  // Import transfer functions:
  if (importTF)
  {
    bool error=false;
    vvVolDesc* vdTF;    // VD for the transfer functions

    vdTF = new vvVolDesc(importFile);
    fio = new vvFileIO();
    switch (fio->loadVolumeData(vdTF, vvFileIO::TRANSFER))
    {
      case vvFileIO::OK:
        vd->tf.copy(&vd->tf._widgets, &vdTF->tf._widgets);
        cerr << "Transfer function imported from: " << importFile << endl;
        break;
      case vvFileIO::FILE_NOT_FOUND:
        cerr << "Transfer functions file not found: " << importFile << endl;
        error = true;
        break;
      default:
        cerr << "Cannot load transfer functions file: " << importFile << endl;
        error = true;
        break;
    }
    delete fio;
    if (error) return false;
  }
  return true;
}

//----------------------------------------------------------------------------
/** Make data modifications for each input file.
  This covers mostly the modifications which could considerably
  change the volume data size and thus should be done as early
  as possible in the conversion process.
  @param v volume description to which to apply the modifications
*/
void vvConv::modifyInputFile(vvVolDesc* v)
{
  if (setRange)
  {
    cerr << "Setting physical data value range." << endl;
    v->real[0] = newRange[0];
    v->real[1] = newRange[1];
  }
  if (autoRealRange)
  {
    float scalarMin, scalarMax;
    cerr << "Automatically setting physical data value range." << endl;
    v->findMinMax(0, scalarMin, scalarMax);
    v->real[0] = scalarMin;
    v->real[1] = scalarMax;
  }
  if (swap)
  {
    cerr << "Swapping bytes." << endl;
    v->toggleEndianness();
  }
  if (sign)
  {
    cerr << "Toggle sign." << endl;
    v->toggleSign();
  }
  if (crop)
  {
    cerr << "Cropping data." << endl;
    v->crop(cropPos[0], cropPos[1], cropPos[2], cropSize[0], cropSize[1], cropSize[2]);
  }
  if (resize)
  {
    cerr << "Resizing data: ";
    if (resizeFactor>0.0f)    // scaling or resizing?
    {
      newSize[0] = (int)(resizeFactor * (float)v->vox[0]);
      newSize[1] = (int)(resizeFactor * (float)v->vox[1]);
      newSize[2] = (int)(resizeFactor * (float)v->vox[2]);
    }
    v->resize(newSize[0], newSize[1], newSize[2], ipt, true);
    cerr << endl;
  }
  if (bpchan>-1)
  {
    cerr << "Converting bytes per channel: ";
    v->convertBPC(bpchan, true);
    cerr << endl;
  }
  if (channels>-1)
  {
    cerr << "Changing number of channels: ";
    v->convertChannels(channels, true);
    cerr << endl;
  }
  if (extractChannel)
  {
    cerr << "Extracting 4th channel: ";
    v->extractChannel(extract, true);
    cerr << endl;
  }
}

//----------------------------------------------------------------------------
/** Make data modifications for the output file.
  This covers the modifications which do not necessarily
  change the volume data size.
*/
void vvConv::modifyOutputFile(vvVolDesc* v)
{
  int i;

  if (invertVoxelOrder)
  {
    cerr << "Inverting voxel order" << endl;
    v->convertVoxelOrder();
  }
  if (swapChannels)
  {
    cerr << "Swapping channels." << endl;
    v->swapChannels(swapChan[0], swapChan[1]);
  }
  if (sphere)
  {
    cerr << "Converting to sphere: ";
    v->makeSphere(outer, inner, ipt, true);
    cerr << endl;
  }
  if (heightField)
  {
    cerr << "Converting to height field: ";
    v->makeHeightField(hfHeight, hfMode, true);
    cerr << endl;
  }
  if (rotate)
  {
    cerr << "Rotating data." << endl;
    v->rotate(rotAxis, rotDir);
  }
  if (flip)
  {
    cerr << "Flipping data." << endl;
    v->flip(flipAxis);
  }
  if (setDist)
  {
    cerr << "Setting distance values." << endl;
    for (i=0; i<3; ++i)
      v->dist[i] = newDist[i];
  }
  if (setPos)
  {
    cerr << "Setting position values." << endl;
    for (i=0; i<3; ++i)
      v->pos[i] = posObj[i];
  }
  if (shift)
  {
    cerr << "Shifting data." << endl;
    v->shift(shiftDist[0], shiftDist[1], shiftDist[2]);
  }
  if (bitshift)
  {
    cerr << "Bit shifting data by " << bshiftDist << " bits: ";
    v->bitShiftData(bshiftDist, true);
    cerr << endl;
  }
  if (drawBox)
  {
    uchar* col;
    col = new uchar[v->bpc];
    int i;
    for (i=0; i<v->bpc; ++i)
    {
      col[i] = uchar(boxColor);
    }
    cerr << "Drawing box." << endl;
    v->drawBox(boxStart[0], boxStart[1], boxStart[2], 
      boxEnd[0], boxEnd[1], boxEnd[2], 0, col);
    delete[] col;
  }
  if (drawLine)
  {
    if (v->bpc==1 && v->chan==1)
    {
      uchar col = uchar(lineColor);
      cerr << "Drawing line." << endl;
      v->drawLine(lineStart[0], lineStart[1], lineStart[2], 
        lineEnd[0], lineEnd[1], lineEnd[2], &col);
    }
    else cerr << "Can draw lines only in 1 byte per voxel datasets." << endl;
  }
  if (croptime)
  {
    v->cropTimesteps(cropSteps[0], cropSteps[1]);
  }
  if (removeTF)
  {
    v->tf._widgets.removeAll();
  }
  if (fillRange)
  {
    v->expandDataRange(true);
  }
  if (zoomData)
  {
    cerr << "Zooming data: ";
    float maxScalar = v->getValueRange();
    zoomRange[1] = ts_clamp(zoomRange[1], 0, int(maxScalar)-1);
    zoomRange[2] = ts_clamp(zoomRange[2], 0, int(maxScalar)-1);
    v->zoomDataRange(zoomRange[0], zoomRange[1], zoomRange[2], true);
  }
  if (animTime>0.0f)
  {
    v->dt = animTime;
  }
  if (deinterlace)
  {
    v->deinterlace();
  }
  if (blend)
  {
    vvVolDesc* blendVD;
    vvFileIO* fio;
    fio = new vvFileIO();
    blendVD = new vvVolDesc(blendFile);
    fio->loadVolumeData(blendVD);
    cerr << "Blending data: ";
    v->blend(blendVD, blendType, true);    
    delete fio;
    delete blendVD;
  }
  if (makeIcon) // make icon from data set
  {
    cerr << "Making icon from slices" << endl;
    vd->makeIcon(makeIconSize);
  }
  if (setIcon)  // load icon from file
  {
    cerr << "Setting icon from file: " << iconFile << endl;
    vvVolDesc* iconVD = new vvVolDesc(iconFile);
    vvFileIO* fio = new vvFileIO();
    if (fio->loadVolumeData(iconVD) != vvFileIO::OK)
    {
      cerr << "Cannot load icon file: " << iconFile << endl;
    }
    else
    {  
      vd->iconSize = ts_min(iconVD->vox[0], iconVD->vox[1]);
      uchar* raw = iconVD->getRaw();
      delete[] vd->iconData;
      vd->iconData = new uchar[vd->iconSize * vd->iconSize * vvVolDesc::ICON_BPP];
      vvToolshed::resample(raw, iconVD->vox[0], iconVD->vox[1], iconVD->bpc * iconVD->chan, 
        vd->iconData, vd->iconSize, vd->iconSize, vvVolDesc::ICON_BPP);
    }
    delete fio;
    delete iconVD;
  }
  if (addChannel)
  {
    cerr << "Adding channel(s) from file: " << addFile << endl;
    vvVolDesc* addVD = new vvVolDesc(addFile);
    vvFileIO* fio = new vvFileIO();
    if (fio->loadVolumeData(addVD) != vvFileIO::OK)
    {
      cerr << "Error: Cannot load file to add." << endl;
    }
    else
    {  
      vd->merge(addVD, vvVolDesc::VV_MERGE_CHAN2VOL);
    }
    delete fio;
    delete addVD;
  }
}

//----------------------------------------------------------------------------
/** Writes the volume data.
  The volume data from the class attribute 'vd' is stored to the
  file name passed on the command line.
  @return true if ok, false on error
*/
bool vvConv::writeVolumeData()
{
  vvFileIO* fio;
  bool ok = true;

  if (vd->frames<1) 
  {
    cerr << "Cannot save file: Destination volume is empty." << endl;
    return false;
  }

  if (leicaRename)
  {
    char* temp = new char[strlen(dstFile) - 3];
    strncpy(temp, dstFile, strlen(dstFile) - 4);
    temp[strlen(dstFile) - 4] = '\0';

    string str = temp;

    assert(lineAverage >= 0 && lineAverage <= 16);
    str += "-L";
    if (lineAverage < 10)
      str += "0";
    {    
      std::stringstream ss;
      ss << lineAverage;
      str += ss.str();
    }

    int isections = int(thickness * 100.0f);
    float fsections = (float)(isections) / 100;
    str += "-S";
    if (fsections < 10)
      str += "0";
    {
      std::stringstream ss;
      ss << fsections;
      str += ss.str();
    }
    if ((int)(fsections * 100) % 10 == 0)
      str+= "0";
    

    str += "-N";
    if (sections < 10)
      str += "0";
    {    
      std::stringstream ss;
      ss << sections;
      str += ss.str();
    }

    str += "-P";
    if (pinhole < 10)
      str += "0";
    {    
      std::stringstream ss;
      ss << pinhole;
      str += ss.str();
    }
   
    str += "-R";
    if (width < 1000 && width >= 100)
      str += "0";
    else if (width < 100)
      str += "00";
    {
      std::stringstream ss;
      ss << width;
      str += ss.str();
    }   
    str += "x";
    if (height < 1000 && height >= 100)
      str += "0";
    else if (height < 100)
      str += "00";
    {
      std::stringstream ss;
      ss << height;
      str += ss.str();
    }   
   
    str += "-Laser";
    for (int i = 0; i < lasercount; i++)
    {
      {
        std::stringstream ss;
        str += "-l";
        ss << laser[i];
        str += ss.str();
      }
      {
        std::stringstream ss;
        str += "-i";
        ss << intensity[i];
        str += ss.str();
      }
      { 
        std::stringstream ss;
        str += "-o";
        ss << offset[i];
        str += ss.str();
      }
      {      
        std::stringstream ss;
        str += "-g";
        ss << gain[i];
        str += ss.str();
      }
    }
  
    str += "-T";
    int totaltime = endtime - starttime;
    int h = 0, m = 0, s = 0;
    h = totaltime / 3600;
    totaltime -= h * 3600;
    m = totaltime / 60;
    totaltime -= m * 60;
    s = totaltime;               
    {
      std::stringstream ss;
      ss << h;
      if (h < 10)
        str += "0";
      str += ss.str();
      str += "h";
    }
    {
      std::stringstream ss;
      ss << m;
      if (m < 10)
        str += "0";
      str += ss.str();
      str += "m";
    }
    {
      std::stringstream ss;    
      ss << s;
      if (s < 10)
        str += "0";
      str += ss.str();
      str += "s";
    }
   
      
    char* finalDst = (char*)(str.c_str());
    strcat(finalDst, ".xvf");

    vd->setFilename(finalDst);
  }
  else
    vd->setFilename(dstFile);
  vd->printInfoLine("Writing: ");
  fio = new vvFileIO();
  fio->setCompression(compression);
  switch (fio->saveVolumeData(vd, overwrite))
  {
    case vvFileIO::OK:
      cerr << "Volume saved successfully." << endl;
      break;
    case vvFileIO::PARAM_ERROR:
      cerr << "Cannot save volume: unknown extension or invalid file name." << endl;
      ok = false;
      break;
    case vvFileIO::FILE_EXISTS:
      cerr << "Cannot overwrite existing file. Use -over parameter." << endl;
      ok = false;
      break;
    default: 
      cerr << "Cannot save volume." << endl; 
      ok = false;
      break;
  }
  delete fio;
  return ok;
}

//----------------------------------------------------------------------------
/** Parse command line arguments.
  @param argc,argv command line arguments
  @return true if parsing ok, false on error
*/
bool vvConv::parseCommandLine(int argc, char** argv)
{
  int arg;    // index of currently processed command line argument
  int i;

  // Parse command line options:
  arg = 0;
  for (;;)
  {
    if ((++arg)>=argc) return true;

    else if (vvToolshed::strCompare(argv[arg], "-h")==0 ||
        vvToolshed::strCompare(argv[arg], "-?")==0 ||
        vvToolshed::strCompare(argv[arg], "/?")==0 ||
        vvToolshed::strCompare(argv[arg], "-help")==0)
    {
      showHelp = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-stat")==0)
    {
      statistics = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-hist")==0)
    {
      histogram = true;
      if ((++arg)>=argc) 
      {
        cerr << "Histogram type expected." << endl;
        return false;
      }
      histType = atoi(argv[arg]);
      if (histType<0 || histType>2)
      {
        cerr << "Invalid histogram type." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-info")==0 ||
        vvToolshed::strCompare(argv[arg], "-i")==0)
    {
      fileInfo = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-fillrange")==0)
    {
      fillRange = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-dicomrename")==0)
    {
      dicomRename = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-leicarename")==0)
    {
      leicaRename = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-deinterlace")==0)
    {
      deinterlace = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-nocompress")==0)
    {
      compression = false;
    }

    else if (vvToolshed::strCompare(argv[arg], "-time")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Time step duration missing." << endl;
        return false;
      }
      animTime = float(atof(argv[arg]));
      if (animTime<=0.0f)
      {
        cerr << "Animation time must be greater than zero (you entered " << animTime << ")." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-files")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Number of files missing." << endl;
        return false;
      }
      files = atoi(argv[arg]);
      if (files<0)
      {
        cerr << "Invalid number of files." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-increment")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "File increment missing." << endl;
        return false;
      }
      increment = atoi(argv[arg]);
      if (files<1)
      {
        cerr << "Invalid increment." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-channels")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Number of channels missing." << endl;
        return false;
      }
      channels = atoi(argv[arg]);
      if (channels<1)
      {
        cerr << "Invalid number of channels." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-over")==0 ||
             vvToolshed::strCompare(argv[arg], "-o")==0)
    {
      overwrite = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-autodetectrealrange")==0)
    {
      autoRealRange = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-invertorder")==0)
    {
      invertVoxelOrder = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-bpc")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Bytes per channel missing." << endl;
        return false;
      }
      bpchan = atoi(argv[arg]);
      if (bpchan!=1 && bpchan!=2 && bpchan!=4)
      {
        cerr << "Number of bytes must be 1, 2, or 4." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-makesphere")==0)
    {
      sphere = true;
      if ((++arg)>=argc) 
      {
        cerr << "Inner sphere diameter missing." << endl;
        return false;
      }
      outer = atoi(argv[arg]);
      if (outer<=0)
      {
        cerr << "Invalid inner sphere diameter." << endl;
        return false;
      }
      if ((++arg)>=argc) 
      {
        cerr << "Outer sphere diameter missing." << endl;
        return false;
      }
      inner = atoi(argv[arg]);
      if (inner<0 || inner>outer)
      {
        cerr << "Invalid outer sphere diameter." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-heightfield")==0)
    {
      heightField = true;
      if ((++arg)>=argc) 
      {
        cerr << "Volume height missing." << endl;
        return false;
      }
      hfHeight = atoi(argv[arg]);
      if (hfHeight<=0)
      {
        cerr << "Height must be greater than zero." << endl;
        return false;
      }
      if ((++arg)>=argc) 
      {
        cerr << "Computation mode missing." << endl;
        return false;
      }
      hfMode = atoi(argv[arg]);
      if (hfMode<0 || hfMode>1)
      {
        cerr << "Invalid computation mode." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-resize")==0)
    {
      resize = true;
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Resize parameter missing." << endl;
          return false;
        }
        newSize[i] = atoi(argv[arg]);
        if (newSize[i]<=0)
        {
          cerr << "Invalid resize parameter." << endl;
          return false;
        }
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-extractchannel")==0)
    {
      extractChannel = true;
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Extract channel parameter missing." << endl;
          return false;
        }
        extract[i] = float(atof(argv[arg]));
        if (extract[i] < 0.0f)
        {
          cerr << "Invalid extraction parameter." << endl;
          return false;
        }
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-scale")==0)
    {
      resize = true;
      if ((++arg)>=argc) 
      {
        cerr << "Scale factor missing." << endl;
        return false;
      }
      resizeFactor = (float)atof(argv[arg]);
      if (resizeFactor <= 0.0f)
      {
        cerr << "Invalid scale factor." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-interpolation")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Interpolation type missing." << endl;
        return false;
      }
      switch (tolower(argv[arg][0]))
      {
        case 'n': ipt = vvVolDesc::NEAREST; break;
        case 't': ipt = vvVolDesc::TRILINEAR; break;
        default: cerr << "Invalid interpolation type." << endl; return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-flip")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Flip axis missing." << endl;
        return false;
      }
      flip = true;
      switch (tolower(argv[arg][0]))
      {
        case 'x': flipAxis = vvVolDesc::X_AXIS; break;
        case 'y': flipAxis = vvVolDesc::Y_AXIS; break;
        case 'z': flipAxis = vvVolDesc::Z_AXIS; break;
        default: cerr << "Invalid flip parameter." << endl; return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-rotate")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Rotation parameter missing." << endl;
        return false;
      }
      rotate = true;
      switch (argv[arg][0])
      {
        case '-': rotDir =-1; break;
        case '+': rotDir = 1; break;
        default: rotDir = 0;
      }
      if (rotDir!=0)
      {
        switch (tolower(argv[arg][1]))
        {
          case 'x': rotAxis = vvVolDesc::X_AXIS; break;
          case 'y': rotAxis = vvVolDesc::Y_AXIS; break;
          case 'z': rotAxis = vvVolDesc::Z_AXIS; break;
          default: rotDir = 0; break;
        }
      }
      if (rotDir==0)
      {
        cerr << "Invalid rotation parameter." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-dist")==0)
    {
      setDist = true;
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Distance parameter missing." << endl;
          return false;
        }
        newDist[i] = (float)atof(argv[arg]);
        if (newDist[i]<=0.0f) 
        {
          cerr << "Invalid distance parameter." << endl;
          return false;
        }
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-realrange")==0)
    {
      setRange = true;
      for (i=0; i<2; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Range parameter missing." << endl;
          return false;
        }
        newRange[i] = (float)atof(argv[arg]);
      }
      if (newRange[0] >= newRange[1]) 
      {
        cerr << "Invalid range parameters. Minimum must be less than maximum." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-crop")==0)
    {
      crop = true;
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Crop position missing." << endl;
          return false;
        }
        cropPos[i] = atoi(argv[arg]);
        if (cropPos[i] < 0) 
        {
          cerr << "Invalid crop position." << endl;
          return false;
        }
      }
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Crop size missing." << endl;
          return false;
        }
        cropSize[i] = atoi(argv[arg]);
        if (cropSize[i]<=0)
        {
          cerr << "Invalid crop size." << endl;
          return false;
        }
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-croptime")==0)
    {
      croptime = true;
      for (i=0; i<2; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Croptime requires two parameters." << endl;
          return false;
        }
        cropSteps[i] = atoi(argv[arg]);
        if (cropSteps[i] < 0) 
        {
          cerr << "Invalid croptime parameter." << endl;
          return false;
        }
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-drawline")==0)
    {
      drawLine = true;
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Line start point missing." << endl;
          return false;
        }
        lineStart[i] = atoi(argv[arg]);
        if (lineStart[i] < 0) 
        {
          cerr << "Invalid line start point." << endl;
          return false;
        }
      }
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Line end point missing." << endl;
          return false;
        }
        lineEnd[i] = atoi(argv[arg]);
        if (lineEnd[i]<=0)
        {
          cerr << "Invalid line end point." << endl;
          return false;
        }
      }
      if ((++arg)>=argc) 
      {
        cerr << "Line color missing." << endl;
        return false;
      }
      lineColor= atoi(argv[arg]);
      if (lineColor<0 || lineColor>255)
      {
        cerr << "Invalid line color. Must be in [0..255]" << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-drawbox")==0)
    {
      drawBox = true;
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Box start point missing." << endl;
          return false;
        }
        boxStart[i] = atoi(argv[arg]);
        if (boxStart[i] < 0) 
        {
          cerr << "Invalid box start point." << endl;
          return false;
        }
      }
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Box end point missing." << endl;
          return false;
        }
        boxEnd[i] = atoi(argv[arg]);
        if (boxEnd[i]<=0)
        {
          cerr << "Invalid box end point." << endl;
          return false;
        }
      }
      if ((++arg)>=argc) 
      {
        cerr << "Box color missing." << endl;
        return false;
      }
      boxColor= atoi(argv[arg]);
      if (boxColor<0 || boxColor>255)
      {
        cerr << "Invalid box color. Must be in [0..255]" << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-shift")==0)
    {
      shift = true;
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Shift amount missing." << endl;
          return false;
        }
        shiftDist[i] = atoi(argv[arg]);
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-debug")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Debug level missing." << endl;
        return false;
      }
      int level = atoi(argv[arg]);
      if (level>=0 && level<=3) 
        vvDebugMsg::setDebugLevel(level);
      else
      {
        cerr << "Invalid debug level." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-swap")==0)
    {
      swap = true;
    }
    
    else if (vvToolshed::strCompare(argv[arg], "-swapchannels")==0)
    {
      swapChannels = true;
      if ((++arg)>=argc) 
      {
        cerr << "First swap channel missing." << endl;
        return false;
      }
      swapChan[0] = atoi(argv[arg]) - 1;
      if (swapChan[0]<0)
      {
        cerr << "Invalid first swap channel." << endl;
        return false;
      }
      if ((++arg)>=argc) 
      {
        cerr << "Second swap channel missing." << endl;
        return false;
      }
      swapChan[1] = atoi(argv[arg]) - 1;
      if (swapChan[1]<0)
      {
        cerr << "Invalid second swap channel." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-sign")==0)
    {
      sign = true;
    }
    
    else if (vvToolshed::strCompare(argv[arg], "-bitshift")==0)
    {
      bitshift = true;
      if ((++arg)>=argc) 
      {
        cerr << "Bit shift amount missing." << endl;
        return false;
      }
      bshiftDist = atoi(argv[arg]);
    }

    else if (vvToolshed::strCompare(argv[arg], "-transfunc")==0)
    {
      importTF = true;
      if ((++arg)>=argc) 
      {
        cerr << "File name for transfer functions import missing." << endl;
        return false;
      }
      importFile = argv[arg];
      if (!vvToolshed::isFile(importFile))  
      {
        cerr << "Transfer functions file not found: " << importFile << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-addchannel")==0)
    {
      addChannel = true;
      if ((++arg)>=argc) 
      {
        cerr << "File with additional channel(s) to add missing." << endl;
        return false;
      }
      addFile = argv[arg];
      if (!vvToolshed::isFile(addFile))  
      {
        cerr << "Additional channel file not found: " << addFile << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-blend")==0)
    {
      blend = true;
      if ((++arg)>=argc) 
      {
        cerr << "File name for blend file missing." << endl;
        return false;
      }
      blendFile = argv[arg];
      if (!vvToolshed::isFile(blendFile))  
      {
        cerr << "Blend file not found: " << blendFile << endl;
        return false;
      }
      if ((++arg)>=argc) 
      {
        cerr << "Blending type missing." << endl;
        return false;
      }
      blendType = atoi(argv[arg]);
    }

    else if (vvToolshed::strCompare(argv[arg], "-mergetype")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Merge type missing." << endl;
        return false;
      }
      if (strcmp(argv[arg], "volume")==0)
      {
        mergeType = 1;
      }
      else if (strcmp(argv[arg], "anim")==0)
      {
        mergeType = 2;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-removetf")==0)
    {
      removeTF = true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-zoomdata")==0)
    {
      zoomData = true;
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Zoom range needs three parameters." << endl;
          return false;
        }
        zoomRange[i] = atoi(argv[arg]);
        if (i==0) 
        {
          if (zoomRange[i] == 0) 
          {
            cerr << "Invalid channel: first channel is 1" << endl;
            return false;
          }
          else if (zoomRange[i] != -1) --zoomRange[i];  // internally treat channel 1 as channel 0
        }
        else if (zoomRange[i] < 0 || zoomRange[i] > 65535) 
        {
          cerr << "Invalid zoom range parameter: must be between 0 and 65535." << endl;
          return false;
        }
      }
      if (zoomRange[1] > zoomRange[2])
      {
        cerr << "Invalid order of zoom range limits. Give low boundary first." << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-loadraw")==0)
    {
      loadRaw = true;
      if ((++arg)>=argc) 
      {
        cerr << "Volume width missing." << endl;
        return false;
      }
      rawWidth = atoi(argv[arg]);
      if ((++arg)>=argc) 
      {
        cerr << "Volume height missing." << endl;
        return false;
      }
      rawHeight = atoi(argv[arg]);
      if ((++arg)>=argc) 
      {
        cerr << "Number of volume slices missing." << endl;
        return false;
      }
      rawSlices = atoi(argv[arg]);
      if ((++arg)>=argc) 
      {
        cerr << "Byte per channel missing." << endl;
        return false;
      }
      rawBPC = atoi(argv[arg]);
      if ((++arg)>=argc) 
      {
        cerr << "Channels missing." << endl;
        return false;
      }
      rawCh = atoi(argv[arg]);
      if ((++arg)>=argc) 
      {
        cerr << "Skip value missing." << endl;
        return false;
      }
      rawSkip = atoi(argv[arg]);
    }

    else if (vvToolshed::strCompare(argv[arg], "-loadcpt")==0)
    {
      loadCPT = true;
      if ((++arg)>=argc) 
      {
        cerr << "CPT volume size missing." << endl;
        return false;
      }
      cptSize = atoi(argv[arg]);
      if ((++arg)>=argc) 
      {
        cerr << "CPT scalar parameter missing." << endl;
        return false;
      }
      cptParam = atoi(argv[arg]);
      if ((++arg)>=argc) 
      {
        cerr << "CPT global min/max specifier missing." << endl;
        return false;
      }
      cptGlobal = (atoi(argv[arg])==0) ? false : true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-loadxb7")==0)
    {
      loadXB7 = true;
      if ((++arg)>=argc) 
      {
        cerr << "XB7 volume size missing." << endl;
        return false;
      }
      xb7Size = atoi(argv[arg]);
      if ((++arg)>=argc) 
      {
        cerr << "XB7 scalar parameter missing." << endl;
        return false;
      }
      xb7Param = atoi(argv[arg]);
      if ((++arg)>=argc) 
      {
        cerr << "XB7 global min/max specifier missing." << endl;
        return false;
      }
      xb7Global = (atoi(argv[arg])==0) ? false : true;
    }

    else if (vvToolshed::strCompare(argv[arg], "-pos")==0)
    {
      setPos = true;
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Position parameter missing." << endl;
          return false;
        }
        posObj[i] = (float)atof(argv[arg]);
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-makeicon")==0)
    {
      makeIcon = true;
      if ((++arg)>=argc) 
      {
        cerr << "Size of new icon missing." << endl;
        return false;
      }
      makeIconSize = atoi(argv[arg]);
    }
    
    else if (vvToolshed::strCompare(argv[arg], "-makevolume")==0)
    {
      if ((++arg)>=argc) 
      {
        cerr << "Creation algorithm missing." << endl;
        return false;
      }
      makeVolume = atoi(argv[arg]);
      for (i=0; i<3; ++i)
      {
        if ((++arg)>=argc) 
        {
          cerr << "Volume size." << endl;
          return false;
        }
        makeVolumeSize[i] = atoi(argv[arg]);
      }
    }
    
    else if (vvToolshed::strCompare(argv[arg], "-seticon")==0)
    {
      setIcon = true;
      if ((++arg)>=argc) 
      {
        cerr << "File name for icon image missing." << endl;
        return false;
      }
      iconFile = argv[arg];
      if (!vvToolshed::isFile(iconFile))  
      {
        cerr << "Icon file not found: " << iconFile << endl;
        return false;
      }
    }

    else if (vvToolshed::strCompare(argv[arg], "-geticon")==0)
    {
      getIcon = true;
    }

    else  // assume filename
    {
      if (argv[arg]==NULL) 
      {
        cerr << "Invalid command line option." << endl;
        return false;
      }
      if (argv[arg][0]=='-')
      {
        cerr << "Invalid command line option: " << argv[arg] << endl;
        return false;
      }
      if (srcFile==NULL)  // parse source file name?
      {
        srcFile = argv[arg];
      }
      else if (dstFile==NULL)      // parse destination file name
      {
        dstFile = argv[arg]; 
      }
      else
      {
        cerr << "Cannot parse command line parameter: " << argv[arg] << endl;
        return false;
      }
    }
  }
}

//----------------------------------------------------------------------------
/// Display command usage help on the command line.
void vvConv::displayHelpInfo()
{
  cerr << "VConv is a command line utility to convert volume files between" << endl;
  cerr << "different file and/or data formats. It is part of the Virvo volume" << endl;
  cerr << "rendering system, which has been developed at the University of Stuttgart," << endl;
  cerr << "Brown University, and UCSD." << endl;
  cerr << "For more information see http://www.calit2.net/~jschulze/" << endl;
  cerr << endl;
  cerr << "Syntax:" << endl;
  cerr << endl;
  cerr << "vconv <source_file.ext> [<destination_file.ext> [<options>]]" << endl;
  cerr << endl;
  cerr << "Parameters:" << endl;
  cerr << endl;
  cerr << "<source_file.ext>" << endl;
  cerr << "Source file name including extension." << endl;
  cerr << "If this is the only parameter, file information is printed." << endl;
  cerr << "The following file types are accepted:" << endl;
  cerr << "rvf                = Raw Volume File (2 x 3 byte header, 8 bit per voxel)" << endl;
  cerr << "xvf                = Extended Volume File" << endl;
  cerr << "avf                = ASCII Volume File" << endl;
  cerr << "tif, tiff          = 2D/3D TIF File" << endl;
  cerr << "dat                = Raw volume data (no header) - automatic format detection" << endl;
  cerr << "rgb                = RGB image file (SGI 8 bit grayscale only)" << endl;
  cerr << "fre, fro           = Visible Human CT data file" << endl;
  cerr << "hdr                = MeshViewer format" << endl;
  cerr << "pd, t1, t2, loc    = Visible Human MRI file" << endl;
  cerr << "raw                = Visible Human RGB file" << endl;
  cerr << "pgm                = Portable Graymap file (P5 binary only)" << endl;
  cerr << "ppm                = Portable Pixmap file (P6 binary only)" << endl;
  cerr << "dcm                = DICOM image file (not all formats)" << endl;
  cerr << "vmr, vtc           = BrainVoyager volume file" << endl;
  cerr << "nrd, nrrd          = Gordon Kindlmann's teem volume file" << endl;
  cerr << "ximg               = General Electric scanner format" << endl;
  cerr << "vis04              = IEEE Visualization 2004 contest format" << endl;
  cerr << "dds                = Microsoft DirectDraw Surface file format " << endl;
  cerr << endl;
  cerr << "<destination_file.ext>" << endl;
  cerr << "Destination file name. The extension determines the destination data format." << endl;
  cerr << "The following file types are accepted:" << endl;
  cerr << "rvf                = Raw Volume File (2 x 3 byte header, 8 bit per voxel)" << endl;
  cerr << "xvf                = Extended Raw Volume File" << endl;
  cerr << "avf                = ASCII Volume File" << endl;
  cerr << "dat                = Raw volume data (no header)" << endl;
  cerr << "tif                = 2D TIF File" << endl;
  cerr << "pgm/ppm            = Density or RGB images (depending on volume data type)" << endl;
  cerr << "nrd                = Gordon Kindlmann's teem volume file" << endl;
  cerr << endl;
  cerr << "<options>" << endl;
  cerr << "The following command line options are accepted in any order." << endl;
  cerr << "The commands in brackets can be used as abbreviations." << endl;
  cerr << endl;
  cerr << "-addchannel <filename>" << endl;
  cerr << " Add all data channels from specified volume file. The following volume" << endl;
  cerr << " parameters must match in both files: width, height, slices, time steps," << endl;
  cerr << " byte per channel." << endl;
  cerr << endl;
  cerr << "-autodetectrealrange" << endl;
  cerr << " Automatically detect the value range of the data values and set realMin/Max" << endl;
  cerr << " accordingly. This parameter is only really useful for float voxels." << endl;
  cerr << " Uses 0..255 for 8 bit and 0..65535 for 16 bit voxels." << endl;
  cerr << endl;
  cerr << "-bitshift <bits>" << endl;
  cerr << " Shift the data of each voxel by <bits> bits, regardless of the data format." << endl;
  cerr << " Negative values shift to the left, positive values shift to the right." << endl;
  cerr << endl;
  cerr << "-blend <filename.xxx> <type>" << endl;
  cerr << " Blend two volume datasets together. The first dataset is the source file," << endl;
  cerr << " the second dataset is passed after the -blend parameter." << endl;
  cerr << " The blending algorithm applies one of the following blending types:" << endl;
  cerr << " 0=average, 1=maximum, 2=minimum" << endl;
  cerr << " The components of each voxel are blended with the given type." << endl;
  cerr << " Both volumes must be of same size and format and have the same voxel" << endl;
  cerr << " distances. The resulting volume will have the same number of frames as" << endl;
  cerr << " the source volume." << endl;
  cerr << " Example: -blend blendfile.xvf 0" << endl;
  cerr << endl;
  cerr << "-bpc <bytes>" << endl;
  cerr << " Set the number of bytes that make up each channel. Supported are:" << endl;
  cerr << " 1 byte for 8 bit data, 2 bytes for 12 or 16 bit data, 4 bytes for float data." << endl;
  cerr << " Conversion strategy: 16 bit values are converted to 8 bit by dropping" << endl;
  cerr << " the least significant bits. Integer values are converted to float by" << endl;
  cerr << " mapping them to [0..1]. Float values are converted to integer by linearly" << endl;
  cerr << " mapping them to [0..maxint] bounded by the min and max float values." << endl;
  cerr << endl;
  cerr << "-channels <num_channels>" << endl;
  cerr << " Change the number of channels to <num_channels>." << endl;
  cerr << " If more than the current number of channels are requested, the new channel" << endl;
  cerr << " will be set to all 0's. If the number of channels is to be reduced, the " << endl;
  cerr << " first <num_channels> channels will remain." << endl;
  cerr << endl;
  cerr << "-crop <pos_x> <pos_y> <pos_z> <width> <height> <slices>" << endl;
  cerr << " Crop a sub-volume from the volume. The crop region starts at the volume" << endl;
  cerr << " position (pos_x|pos_y|pos_z), which is the top-left-front voxel of the" << endl;
  cerr << " crop region. The voxel counts start with 0 and go to size-1. The size of" << endl;
  cerr << " the cropped volume is width * height * slices voxels." << endl;
  cerr << endl;
  cerr << "-croptime <first_time_step> <num_steps>" << endl;
  cerr << " Crop a sub-set of time steps from a volume animation." << endl;
  cerr << " first_time_step is the index of the first time step to use, with 0 being the" << endl;
  cerr << " first time step in the file. num_steps is the number of time steps to crop." << endl;
  cerr << endl;
  cerr << "-deinterlace" << endl;
  cerr << " Corrects a dataset with interlaced slices. The first slice will remain the" << endl;
  cerr << " first slice, the second slice will be taken from halfway into the dataset." << endl;
  cerr << endl;
  cerr << "-dicomrename" << endl;
  cerr << " Rename DICOM files to reflect sequence number and slice location." << endl;
  cerr << " The source files must be named in ascending order, e.g., 'file001.dcm'," << endl;
  cerr << " 'file002.dcm'..." << endl;
  cerr << " After processing this command, the files will be called 'seq001-loc000001.dcm'" << endl;
  cerr << " etc." << endl;
  cerr << endl;
  cerr << "-leicarename" << endl;
  cerr << " Rename Leica files." << endl;
  cerr << endl;
  cerr << "-dist <dx> <dy> <dz>" << endl;
  cerr << " Set the voxel sampling distance [mm]. This affects only the voxel 'shape'," << endl;
  cerr << " the voxel data remain untouched." << endl;
  cerr << endl;
  cerr << "-drawbox <start_x> <start_y> <start_z> <end_x> <end_y> <end_z> <value>" << endl;
  cerr << " Draw a solid 3D box into all animation steps of the dataset." << endl;
  cerr << " The box starts at position (start_x|start_y|start_z) and extends to position" << endl;
  cerr << " (end_x|end_y|end_z) The scalar value of the voxels in the box will be <value>." << endl;
  cerr << endl;
  cerr << "-drawline <start_x> <start_y> <start_z> <end_x> <end_y> <end_z> <value>" << endl;
  cerr << " Draw a 3D line into all animation steps of the dataset." << endl;
  cerr << " The line starts at position (start_x|start_y|start_z) and extends to position" << endl;
  cerr << " (end_x|end_y|end_z) The scalar value of the voxels on the line will be <value>." << endl;
  cerr << endl;
  cerr << "-extractchannel <r> <g> <b>" << endl;
  cerr << " Extract a channel from an RGB data set and make it the 4th channel." << endl;
  cerr << " r,g,b indicate the color component weights that characterize the channel." << endl;
  cerr << " The affected RGB values that make up the 4th channel are set to zero." << endl;
  cerr << " Example: -extractchannel 1 0.5 0" << endl;
  cerr << " The example sets the 4th channel value for all voxels in which green is half" << endl;
  cerr << " of red and blue is zero. Example: R=100, G=50, B=0 => channel#4=100" << endl;
  cerr << endl;
  cerr << "-files <num>" << endl;
  cerr << " Use <num> files to create a volume from slices or a volume animation" << endl;
  cerr << " from volume files. Use num=0 to load as many files" << endl;
  cerr << " as present in continuous order." << endl;
  cerr << " The source file will be the first animation frame." << endl;
  cerr << " The filename numbering must be right before the suffix. The numbering" << endl;
  cerr << " must include leading zeroes." << endl;
  cerr << " Example: file001.rvf, file002.rvf, ..." << endl;
  cerr << endl;
  cerr << "-fillrange" << endl;
  cerr << " Expand data range to occupy the entire value range. This is useful to take" << endl;
  cerr << " advantage of the entire possible value range. For example, if the scanned" << endl;
  cerr << " 16 bit data occupy only values between 50 and 1000, they will be mapped to" << endl;
  cerr << " a range of 0 to 65535." << endl;
  cerr << endl;
  cerr << "-flip <x|y|z>" << endl;
  cerr << " Flip volume data along a coordinate axis." << endl;
  cerr << endl;
  cerr << "-geticon" << endl;
  cerr << " Write the icon to TIF file of same name as volume." << endl;
  cerr << endl;
  cerr << "-heightfield <height> <mode>" << endl;
  cerr << " Computes height field from 2D image. Source data set must be single slice." << endl;
  cerr << " Works with any number of bytes per voxel." << endl;
  cerr << " <height> is number of slices destination volume has." << endl;
  cerr << " <mode> determines how space below height surface is filled:" << endl;
  cerr << " 0 for empty (using lowest values in data range), 1 for same values as height." << endl;
  cerr << endl;
  cerr << "-help (-h)" << endl;
  cerr << " Display this help information." << endl;
  cerr << endl;
  cerr << "-hist <type>" << endl;
  cerr << " Creates histogram(s) for the first animation frame. Separate histograms are" << endl;
  cerr << " created for each channel. <type> determines the histogram type:" << endl;
  cerr << " 0 = ASCII numbers" << endl;
  cerr << " 1 = graph as PGM image (image file name = volume file name, logarithmic y axis)" << endl;
  cerr << " 2 = ASCII numbers in files, one file per channel" << endl;
  cerr << " The vertical axis is logarithmic." << endl;
  cerr << endl;
  cerr << "-increment <num>" << endl;
  cerr << " Additional paramenter to -files: defines the step size to calculate the" << endl;
  cerr << " next higher file name. Example: if <num> is 3, then file names will be read" << endl;
  cerr << " in the following order: file001.tif, file004.tif, file007.tif..." << endl;
  cerr << endl;
  cerr << "-info (-i)" << endl;
  cerr << " Display file information about the first file parameter." << endl;
  cerr << " No conversion is done, even if according parameters are passed." << endl;
  cerr << " This command is automatically executed if only one file parameter is passed." << endl;
  cerr << endl;
  cerr << "-interpolation <n|t>" << endl;
  cerr << " Define the type of interpolation to use whenever resampling is necessary." << endl;
  cerr << " The available types are: n=nearest neighbor (default), t=trilinear." << endl;
  cerr << " This parameter affects the resize, scale, and sphere operations." << endl;
  cerr << endl;
  cerr << "-invertorder" << endl;
  cerr << " Invert voxel order: order of voxels and slices will be inverted.";
  cerr << endl;
  cerr << "-loadraw <width> <height> <slices> <bpc> <ch> <skip>" << endl;
  cerr << " Load a non-virvo raw volume data file. The parameters are:" << endl;
  cerr << " <width> <height> <slices> = volume size [voxels]" << endl;
  cerr << " <bpc>                     = bytes per channel" << endl;
  cerr << " <ch>                      = number of channels" << endl;
  cerr << " <skip>                    = number of bytes to skip (to ignore header)" << endl;
  cerr << endl;
  cerr << "-loadcpt <size> <param> <minmax>" << endl;
  cerr << " Load a checkpoint particles file. The parameters are:" << endl;
  cerr << " <size    = volume edge length [voxels]" << endl;
  cerr << " <param>  = index of parameter to use (0=first parameter, -1=speed)" << endl;
  cerr << " <minmax> = 1 to use global min/max values, 0 to use min/max values per frame" << endl;
  cerr << endl;
  cerr << "-loadxb7 <size> <param> <minmax>" << endl;
  cerr << " Load an XB7 particles file. The parameters are:" << endl;
  cerr << " <size    = volume edge length [voxels]" << endl;
  cerr << " <param>  = index of parameter to use (0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz, 6=r, 7=i, 8=speed" << endl;
  cerr << " <minmax> = 1 to use global min/max values, 0 to use min/max values per frame" << endl;
  cerr << endl;
  cerr << "-makeicon <size>" << endl;
  cerr << " Create an icon from the data set. The icon is created by blending all slices" << endl;
  cerr << " of the data set together with maximum intensity projection (MIP)." << endl;
  cerr << endl;
  cerr << "-makesphere <outer_diameter> <inner_diameter>" << endl;
  cerr << " Make a sphere from a volume by projecting it onto a sphere." << endl;
  cerr << " The z=0 plane will be projected to a sphere of diameter <outer_diameter>" << endl;
  cerr << " voxels, the z=z_max plane will be projected to a sphere of diameter" << endl;
  cerr << " <inner_diameter> voxels." << endl;
  cerr << " The area between the spheres will be filled with the source volume data, the" << endl;
  cerr << " height information is mapped linearly to the area between the two bounding" << endl;
  cerr << " spheres. The area inside of the inner sphere will be zero filled." << endl;
  cerr << " The resulting volume is a cube with an edge length of <outer_diameter> voxels." << endl;
  cerr << " The default interpolation method is nearest neighbor. It can be changed with" << endl;
  cerr << " the '-interpolation' parameter." << endl;
  cerr << endl;
  cerr << "-makevolume <algorithm> <width> <height> <slices>" << endl;
  cerr << " Create a volume algorithmically. Algorithm is one of:" << endl;
  cerr << " 0: default Virvo volume, 8 frames" << endl;
  cerr << " 1: top=red, bottom=green, 1 channel, 1 bpc" << endl;
  cerr << " 2: 4 channel test data set" << endl;
  cerr << endl;
  cerr << "-mergetype <type>" << endl;
  cerr << " Merge type for -files parameter:" << endl;
  cerr << " 'volume': make a volume (each file is a slice)" << endl;
  cerr << " 'anim':   make an animation (each file is a time step)" << endl;
  cerr << endl;
  cerr << "-nocompress" << endl;
  cerr << " Suppress data compression when writing xvf files." << endl;
  cerr << endl;
  cerr << "-over (-o)" << endl;
  cerr << " Overwrite destination files." << endl;
  cerr << endl;
  cerr << "-pos <x> <y> <z>" << endl;
  cerr << " Set 3D position of volume origin in object space." << endl;
  cerr << endl;
  cerr << "-removetf" << endl;
  cerr << " Remove all transfer functions from the volume." << endl;
  cerr << endl;
  cerr << "-realrange <min> <max>" << endl;
  cerr << " Set the physical scalar value range as floating point numbers. This only has" << endl;
  cerr << " a meaning for scalar value volumes (8 and 16 bit per voxel)." << endl;
  cerr << " It affects just the display of range information and does not modify the data." << endl;
  cerr << endl;
  cerr << "-resize <width> <height> <slices>" << endl;
  cerr << " Resize the volume to the new size of width * height * slices voxels." << endl;
  cerr << " The default interpolation method is nearest neighbor. It can be changed with" << endl;
  cerr << " the '-interpolation' parameter." << endl;
  cerr << endl;
  cerr << "-rotate <[+|-][x|y|z]>" << endl;
  cerr << " Rotate volume data by 90 degrees about a coordinate axis." << endl;
  cerr << " Example: '-rot +y' rotates the data about the y (vertical) axis by 90" << endl;
  cerr << " degrees to the right (looking at the origin from an arbitrary point on" << endl;
  cerr << " the positive half of the rotation axis)." << endl;
  cerr << endl;
  cerr << "-scale <factor>" << endl;
  cerr << " Scale all edges by <factor>." << endl;
  cerr << " The default interpolation method is nearest neighbor. It can be changed with" << endl;
  cerr << " the '-interpolation' parameter." << endl;
  cerr << endl;
  cerr << "-seticon <filename>" << endl;
  cerr << " Set the icon to the image in <filename>. Alpha channels are used if present." << endl;
  cerr << endl;
  cerr << "-shift <x> <y> <z>" << endl;
  cerr << " Shift the volume along the coordinate axes by the passed voxel values." << endl;
  cerr << " Border values reappear on the opposite side." << endl;
  cerr << endl;
  cerr << "-sign" << endl;
  cerr << " Toggle the sign of the data. Converts unsigned to signed and vice versa." << endl;
  cerr << " Inverts the most significant bit of each scalar value." << endl;
  cerr << endl;
  cerr << "-stat" << endl;
  cerr << " Displays the same as '-info', plus some statistics about the volume data." << endl;
  cerr << endl;
  cerr << "-swap" << endl;
  cerr << " Swap endianness of data bytes. The result depends on the data format:" << endl;
  cerr << " 8 bit scalar values are not affected, for 16 bit voxels high and low byte" << endl;
  cerr << " are swapped, RGB volumes become BGR, RGBA volumes become ABGR." << endl;
  cerr << endl;
  cerr << "-swapchannels <channel1> <channel2>" << endl;
  cerr << " Swap two data channels. <channel1> and <channel2> must be in" << endl;
  cerr << " [1..num_channels]. This command does not work on data sets with only" << endl;
  cerr << " one channel." << endl;
  cerr << endl;
  cerr << "-time <sec>" << endl;
  cerr << " Set the time [seconds] that each animation frame takes to display." << endl;
  cerr << endl;
  cerr << "-transfunc <filename.xvf>" << endl;
  cerr << " Import the transfer functions from a file." << endl;
  cerr << endl;
  cerr << "-zoomdata <channel> <low> <high>" << endl;
  cerr << " Zoom the data of <channel> to the range between the <low> and <high> data" << endl;
  cerr << " values. Example: -zoomdata 1 10 80" << endl;
  cerr << " In the example, the data values from 10 to 80 will be expanded to the" << endl;
  cerr << " full range of 256 (or 65536 for 16 bit channels) values." << endl;
  cerr << " Channel 1 is the first channel, data value 0 is the lowest, 255 the" << endl;
  cerr << " highest value for 8 bit per channel data (65535 for 16 bit)." << endl;
  cerr << endl;
  cerr << "Examples:" << endl;
  cerr << "Display file information:              vconv skull.tiff" << endl;
  cerr << "Convert file type from 3DTIFF to XVF:  vconv skull.tiff skull.xvf" << endl;
  cerr << "Create a volume from 30 image files:   vconv slc001.tif skull.xvf -files 30" << endl;
  cerr << "Create 9 time steps of transient data: vconv step001.tiff anim.xvf -files 9" << endl;
  cerr << "Scale volume down to half edge size:   vconv large.xvf small.xvf -scale 0.5" << endl;
#ifndef WIN32
  cerr << endl;
#endif
}

//----------------------------------------------------------------------------
/** Main routine.
  @param argc,argv command line arguments
  @return 0 if file conversion ok, 1 if an error occurred
*/
int vvConv::run(int argc, char** argv)
{
  int error = 0;

  cerr << "VConv Version " << VV_VERSION << "." << VV_RELEASE << endl;
  cerr << "(C) " << VV_YEAR << " Brown University" << endl;
  cerr << "Author: Jurgen P. Schulze (jschulze@ucsd.edu)" << endl;
  cerr << "VConv comes with ABSOLUTELY NO WARRANTY." << endl;
  cerr << "It is free software, and you are welcome to redistribute it under" << endl;
  cerr << "the LGPL license. See the file 'license.txt' in the program directory." << endl;
  cerr << endl;

  if (argc < 2)
  {
    cerr << "Syntax:" << endl;
    cerr << "  vconv <source_file.ext> [<destination_file.ext>] [<options>]" << endl;
    cerr << endl;
    cerr << "For more information type: vconv -help" << endl;
    cerr << endl;
    cerr << "The following options are supported:" << endl;
    cerr << "-addchannel <filename>             add data channel(s)" << endl;
    cerr << "-autodetectrealrange               set real data range automatically" << endl;
    cerr << "-bitshift <bits>                   shift voxel data" << endl;
    cerr << "-blend <filename> <type>           blend two files together" << endl;
    cerr << "-bpc <bytes>                       set bytes per channel" << endl;
    cerr << "-channels <num_ch>                 change the number of channels" << endl;
    cerr << "-crop <x> <y> <z> <w> <h> <s>      crop volume" << endl;
    cerr << "-croptime <first_step> <num_steps> crop a sequence of time steps" << endl;
    cerr << "-deinterlace                       corrects interlaced slices" << endl;
    cerr << "-dicomrename                       automatically rename dicom files" << endl;
    cerr << "-leicarename                       automatically rename Leica files" << endl;
    cerr << "-dist <dx> <dy> <dz>               set voxel sampling distance" << endl;
    cerr << "-drawbox <a> <b> <c> <d> <e> <f> <v> draw a 3D box of voxels" << endl;
    cerr << "-drawline <a> <b> <c> <d> <e> <f> <v> draw a 3D line of voxels" << endl;
    cerr << "-extractchannel <r> <g> <b>        extract 4th channel from RGB" << endl;
    cerr << "-files <num>                       merge multiple files" << endl;
    cerr << "-fillrange                         expand data range" << endl;
    cerr << "-flip <x|y|z>                      flip volume in axis direction" << endl;
    cerr << "-geticon                           write icon to file" << endl;
    cerr << "-heightfield <height> <mode>       calculate height field from slice" << endl;
    cerr << "-help                              verbose options list" << endl;
    cerr << "-hist <type>                       create histogram" << endl;
    cerr << "-increment <num>                   increment for -files" << endl;
    cerr << "-info                              display information about volume" << endl;
    cerr << "-interpolation <n|t>               set interpolation type" << endl;
    cerr << "-invertorder                       invert voxel order" << endl;
    cerr << "-loadraw <w> <h> <s> <bc> <c> <sk> load raw volume data from file" << endl;
    cerr << "-loadcpt <size> <param> <minmax>   load checkpoint particles file" << endl;
    cerr << "-loadxb7 <size> <param> <minmax>   load XB7 particles file" << endl;
    cerr << "-makeicon <size>                   create icon from data set" << endl;
    cerr << "-makesphere <outer> <inner>        project volume data to a sphere" << endl;
    cerr << "-makevolume <alg> <w> <h> <s>      algorithmically create volume" << endl;
    cerr << "-mergetype <type>                  merge type: 'volume' or 'anim'" << endl;
    cerr << "-nocompress                        suppress compression" << endl;
    cerr << "-over                              overwrite destination file" << endl;
    cerr << "-pos <x> <y> <z>                   set position" << endl;
    cerr << "-realrange <min> <max>             set physical scalar value range" << endl;
    cerr << "-removetf                          remove all transfer functions" << endl;
    cerr << "-resize <w> <h> <s>                resize volume (see -interpolation)" << endl;
    cerr << "-rotate <[+|-][x|y|z]>             rotate volume 90 degrees" << endl;
    cerr << "-scale <factor>                    scale volume (see -interpolation)" << endl;
    cerr << "-seticon <filename>                set the icon image" << endl;
    cerr << "-shift <x> <y> <z>                 shift parallel to a coordinate axis" << endl;
    cerr << "-stat                              display volume data statistics" << endl;
    cerr << "-sign                              toggle sign" << endl;
    cerr << "-swap                              swap endianness of voxel data bytes" << endl;
		cerr << "-swapchannels <ch1> <ch2>          swap two channels in each voxel" << endl;
    cerr << "-time <dt>                         set animation time per frame" << endl;
    cerr << "-transfunc <filename.xvf>          import transfer functions from file" << endl;
    cerr << "-zoomdata <channel> <low> <high>   zoom data range" << endl;
    cerr << endl;

#ifndef WIN32
  cerr << endl;
#endif
    return 0;
  }

  // Now there must be at least two command line arguments, so
  // we can begin the regular parsing process:
  if (!parseCommandLine(argc, argv))  return 1;

  // Check for help parameter:
  if (showHelp || srcFile==NULL)
  { 
    displayHelpInfo();
    return 0;
  }

  // Check if source file exists:
  if (makeVolume==-1 && !vvToolshed::isFile(srcFile))   
  {
    cerr << "Source file not found: " << srcFile << endl;
    return 1;
  }
  
  if (makeVolume>-1) dstFile = srcFile;

  if (histogram)
  {
    cerr << "Reading volume file: " << srcFile << endl;
  	vvVolDesc* tmpVD = new vvVolDesc(srcFile);
    vvFileIO* fio = new vvFileIO();
    error = 0;
    if (fio->loadVolumeData(tmpVD) != vvFileIO::OK)
    {
      cerr << "Cannot load file for histogram." << endl;
      error = 1;
    }
    else if (tmpVD->bpc>2) 
    {
      cerr << "Cannot create histogram for float data type." << endl;
    }
    else if (histType==0)   // ASCII on screen
    {
      cerr << endl;
      for (int m=0; m<tmpVD->chan; ++m)
      {
        if (tmpVD->chan>1) cerr << "Channel " << m << ":" << endl;
        tmpVD->printHistogram(0, m);
        cerr << endl;
      }
    }
    else if (histType==1)   // image file
    {
      char* basePath = new char[strlen(srcFile) + 1];
      vvToolshed::extractBasePath(basePath, srcFile);
      char* imgFileName = new char[strlen(srcFile) + 15];
      for (int m=0; m<tmpVD->chan; ++m)
      {
        if (tmpVD->chan>1)
        {
          sprintf(imgFileName, "%s-hist-ch%02d.ppm", basePath, m);
        }
        else 
        {
          sprintf(imgFileName, "%s-hist.ppm", basePath);
        }
        vvVolDesc* imgVD = new vvVolDesc(imgFileName);
        imgVD->vox[0] = 256;
        imgVD->vox[1] = 256;
        imgVD->vox[2] = 1;
        imgVD->bpc = 1;
        imgVD->chan = 3;
        uchar* imgData = new uchar[imgVD->vox[0] * imgVD->vox[1] * 3];
        int size[2] = {imgVD->vox[0], imgVD->vox[1]};
        vvColor col(1.0f, 1.0f, 1.0f);
        tmpVD->makeHistogramTexture(-1, m, 1, size, imgData, vvVolDesc::VV_LOGARITHMIC, &col, tmpVD->real[0], tmpVD->real[1]);
        imgVD->addFrame(imgData, vvVolDesc::ARRAY_DELETE);
        imgVD->frames = 1;
        imgVD->flip(vvVolDesc::Y_AXIS);
        vvFileIO* fio = new vvFileIO();
        switch (fio->saveVolumeData(imgVD, false))
        {
          case vvFileIO::FILE_EXISTS:
            cerr << "Cannot write histogram: file exists." << endl;
            break;
          case vvFileIO::OK:
            cerr << "Histogram file created: " << imgVD->getFilename() << endl;
            break;
          default:
            cerr << "Cannot write histogram file" << endl;
            break;
        }
        delete imgVD;
        delete fio;
      }
      delete[] imgFileName;
      delete[] basePath;
    }
    else if (histType==2)   // create ASCII histogram and save to file
    {
      cerr << "Creating histogram file(s)" << endl;
      tmpVD->createHistogramFiles();
    }
    else assert(0);
    delete fio;     
    delete tmpVD;
    return error;
  }

  if (statistics)    // volume data statistics
  {
    cerr << "Reading volume file: " << srcFile << endl;
  	vvVolDesc* tmpVD = new vvVolDesc(srcFile);
    vvFileIO* fio = new vvFileIO();
    error = 0;
    if (fio->loadVolumeData(tmpVD) != vvFileIO::OK)
    {
      cerr << "Cannot load file for statistics." << endl;
      error = 1;
    }
    else
    {
      cerr << endl;
      tmpVD->printVolumeInfo();
      tmpVD->printStatistics();
      cerr << endl;
    }
    delete fio;     
    delete tmpVD;
    return error;
  }

  if (getIcon) // extract icon to file
  {
    char* iconFile = new char[strlen(srcFile) + 10];
    char* basePath = new char[strlen(srcFile) + 1];
    vvToolshed::extractBasePath(basePath, srcFile);
    sprintf(iconFile, "%s-icon.tif", basePath);

    cerr << "Extracting icon to file: " << iconFile << endl;
    
    // Load icon from volume file:
    cerr << "Reading volume file: " << srcFile << endl;
  	vvVolDesc* tmpVD = new vvVolDesc(srcFile);
    vvFileIO* fio = new vvFileIO();
    if (fio->loadVolumeData(tmpVD, vvFileIO::ICON) != vvFileIO::OK)
    {
      cerr << "Cannot load file to read icon." << endl;
      error = 1;
      delete tmpVD;
      delete fio;     
      return error;
    }
    delete fio;

    // Create icon volume:
    vvVolDesc* iconVD = new vvVolDesc();
    iconVD->vox[0] = tmpVD->iconSize;
    iconVD->vox[1] = tmpVD->iconSize;
    iconVD->vox[2] = 1;
    iconVD->setFilename(iconFile);
    iconVD->bpc = 1;
    iconVD->chan = vvVolDesc::ICON_BPP;
    uchar* iconData = new uchar[iconVD->getSliceBytes()];
    memcpy(iconData, tmpVD->iconData, iconVD->getSliceBytes());
    iconVD->addFrame(iconData, vvVolDesc::ARRAY_DELETE);
    iconVD->frames = 1;
    delete tmpVD;
    delete[] iconFile;
    delete[] basePath;

    // Save icon volume to file:
    vvFileIO* iconFIO = new vvFileIO();    
    switch (iconFIO->saveVolumeData(iconVD, overwrite))
    {
      case vvFileIO::FILE_EXISTS:
        cerr << "Icon file exists." << endl;
        error = 1;
        break;
      case vvFileIO::OK:
        cerr << "Icon file created." << endl;
        break;
      default:
        cerr << "Cannot write icon file" << endl;
        error = 1;
        break;
    }
    delete iconFIO;
    delete iconVD;
    return error;
  }

  if ((fileInfo || dstFile==NULL) && makeVolume==-1)    // info only mode
  {
    vvFileIO::ErrorType et;
 
    cerr << "Reading header information from file: " << argv[1] << endl;
  	vd = new vvVolDesc(srcFile);
    vvFileIO* fio = new vvFileIO();
    error = 0;
    if ((et=fio->loadVolumeData(vd, vvFileIO::HEADER)) != vvFileIO::OK)
    {
      if (et==vvFileIO::PARAM_ERROR)
        cerr << "Unknown file type." << endl;
      else
        cerr << "Cannot load file header." << endl;
      error = 1;
    }
    else
    {
      cerr << endl;
      vd->printVolumeInfo();
      cerr << endl;                  
    }
    delete fio;     
    delete vd;
    vd = NULL;
    return error;
  }

  if (dicomRename)    // DICOM file rename mode is special
  {
    return renameDicomFiles();
  }

  // Check if destination file exists:
  if (!overwrite && vvToolshed::isFile(dstFile))    
  {
    cerr << "Destination file '" << dstFile << "' exists." << endl;
    cerr << "Select a different file name or use the -over parameter." << endl;
    return 1;
  }

  cerr << "Reading volume data." << endl;
  if (!readVolumeData()) return 1;

  modifyOutputFile(vd);

  cerr << "Writing destination file: " << dstFile << endl;
  if (!writeVolumeData()) return 1;
  cerr << "Done." << endl;
  return 0;
}


//----------------------------------------------------------------------------
/** Rename a bunch of DICOM files according to their information on sequence and slice IDs.
  @return 0 if ok, 1 on error
*/
int vvConv::renameDicomFiles()
{
  char* oldName;
  char newName[256];
  int dcmSeq, dcmSlice;   // DICOM sequence and slice IDs
  float dcmSLoc;
  bool done = false;
  int error = 0;

  oldName = new char[strlen(srcFile) + 1];
  strcpy(oldName, srcFile);
  vvFileIO* fio = new vvFileIO();
  while (!done)
  {
    vd = new vvVolDesc(oldName);
    if (fio->loadDicomFile(vd, &dcmSeq, &dcmSlice, &dcmSLoc) != vvFileIO::OK)
    {
      cerr << "Cannot load Dicom file." << endl;
      error = 1;
      done = true;
    }
    else
    {
      sprintf(newName , "seq%03d-loc%06d.dcm", dcmSeq, int((dcmSLoc + 100.0f) * 1000.0f));
      if (rename(oldName, newName) != 0)
      {
        cerr << "Could not rename " << oldName << " to " << newName << endl;
        done = true;
      }
      else
      {
        cerr << "File " << oldName << " renamed to " << newName << endl;
        if (!vvToolshed::increaseFilename(oldName))
        {
          cerr << "Cannot increase filename '" << oldName << "'." << endl;
          done = true;
        }
        else if (!vvToolshed::isFile(oldName))
          done = true;
      }
    }
    delete vd;
    vd = NULL;
  }
  delete fio;
  delete oldName;
  return error;    
}

//----------------------------------------------------------------------------
/// Main function for the volume converter
int main(int argc, char* argv[])
{
  vvConv* vconv;  
  int error;
  
	vconv = new vvConv();
  error = vconv->run(argc, argv);
  delete vconv;
  return error;
}

//============================================================================
// End of File
//============================================================================
