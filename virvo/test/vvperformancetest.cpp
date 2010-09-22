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

#include "vvperformancetest.h"
#include "../src/vvtoolshed.h"
#include "../src/vvvecmath.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <ctime>
#include <float.h>

using std::cerr;
using std::endl;

#include <sys/stat.h>

#define MAX_LINE_LEN 4096

vvTestResult::vvTestResult()
{

}

vvTestResult::~vvTestResult()
{

}

void vvTestResult::setDiffTimes(const std::vector<float> diffTimes)
{
  _diffTimes = diffTimes;
}

std::vector<float> vvTestResult::getDiffTimes() const
{
  return _diffTimes;
}

float vvTestResult::getTotalTime() const
{
  return _totalTime;
}

float vvTestResult::getAvgTime() const
{
  return _avgTime;
}

float vvTestResult::getVariance() const
{
  return _variance;
}

float vvTestResult::getMaxTime() const
{
  return _maxTime;
}

float vvTestResult::getMinTime() const
{
  return _minTime;
}

void vvTestResult::calc()
{
  _minTime = FLT_MAX;
  _maxTime = -FLT_MAX;
  _totalTime = 0.0f;

  std::vector<float>::const_iterator it;
  for (it = _diffTimes.begin(); it != _diffTimes.end(); ++it)
  {
    const float t = *it;

    if (t > _maxTime)
    {
      _maxTime = t;
    }
    else if (t < _minTime)
    {
      _minTime = t;
    }

    _totalTime += t;
  }
  _avgTime = _totalTime / static_cast<float>(_diffTimes.size());

  // Calc variance.
  _variance = 0.0f;

  for (it = _diffTimes.begin(); it != _diffTimes.end(); ++it)
  {
    float t = *it;

    _variance += (t - _avgTime) * (t - _avgTime);
  }
  _variance /= static_cast<float>(_diffTimes.size());
}

vvPerformanceTest::vvPerformanceTest()
{
  _verbose = true;
  _testResult = new vvTestResult();
  _iterations = 1;
  _frames = 90;
  _quality = 1.0f;
  _geomType = vvTexRend::VV_AUTO;
  _voxelType = vvTexRend::VV_BEST;
  _testAnimation = VV_ROT_Y;
  _projectionType = vvObjView::PERSPECTIVE;
}

vvPerformanceTest::~vvPerformanceTest()
{
  delete _testResult;
}

void vvPerformanceTest::writeResultFiles()
{
#if !defined(_WIN32)
  // Text file with summary.
  char* summaryFile = new char[80];
  time_t now = time(NULL);
  struct tm  *ts;

  ts = localtime(&now);
  strftime(summaryFile, 80, "%Y-%m-%d_%H:%M:%S_%Z_summary.txt", ts);

  FILE* handle = fopen(summaryFile, "w");

  if (handle != NULL)
  {
    _testResult->calc();
    char* dateStr = new char[80];
    strftime(dateStr, 80, "%Y-%m-%d, %H:%M:%S %Z", ts);
    fprintf(handle, "************************* Summary test %i *************************\n", _id);
    fprintf(handle, "Test performed at:....................%s\n", dateStr);
    fprintf(handle, "Total profiling time:.................%f\n", _testResult->getTotalTime());
    fprintf(handle, "Average time per frame:...............%f\n", _testResult->getAvgTime());
    fprintf(handle, "Variance:.............................%f\n", _testResult->getVariance());
    fprintf(handle, "Max rendering time:...................%f\n", _testResult->getMaxTime());
    fprintf(handle, "Min rendering time:...................%f\n", _testResult->getMinTime());
    fclose(handle);
  }

  // Csv file simply with the diff times.
  char* csvFile = new char[80];
  strftime(csvFile, 80, "%Y-%m-%d_%H:%M:%S_%Z_times.csv", ts);

  handle = fopen(csvFile, "w");

  if (handle != NULL)
  {
    std::vector<float> times = _testResult->getDiffTimes();
    std::vector<float>::const_iterator it;

    fprintf(handle, "\"TIME\",\"MODELVIEW_MATRIX\"");
    for (it = times.begin(); it != times.end(); ++it)
    {
      fprintf(handle, "%f\n", *it);
    }
    fclose(handle);
  }
#endif
}

void vvPerformanceTest::setId(const int id)
{
  _id = id;
}

void vvPerformanceTest::setIterations(const int iterations)
{
  _iterations = iterations;
}

void vvPerformanceTest::setVerbose(const bool verbose)
{
  _verbose = verbose;
}

void vvPerformanceTest::setQuality(const float quality)
{
  _quality = quality;
}

void vvPerformanceTest::setBrickDims(const vvVector3& brickDims)
{
  _brickDims = brickDims;
}

void vvPerformanceTest::setBrickDimX(const float brickDimX)
{
  _brickDims[0] = brickDimX;
}

void vvPerformanceTest::setBrickDimY(const float brickDimY)
{
  _brickDims[1] = brickDimY;
}

void vvPerformanceTest::setBrickDimZ(const float brickDimZ)
{
  _brickDims[2] = brickDimZ;
}

void vvPerformanceTest::setGeomType(const vvTexRend::GeometryType geomType)
{
  _geomType = geomType;
}

void vvPerformanceTest::setVoxelType(const vvTexRend::VoxelType voxelType)
{
  _voxelType = voxelType;
}

void vvPerformanceTest::setFrames(const int frames)
{
  _frames = frames;
}

void vvPerformanceTest::setTestAnimation(const TestAnimation testAnimation)
{
  _testAnimation = testAnimation;
}

void vvPerformanceTest::setProjectionType(const vvObjView::ProjectionType projectionType)
{
  _projectionType = projectionType;
}

int vvPerformanceTest::getId() const
{
  return _id;
}

int vvPerformanceTest::getIterations() const
{
  return _iterations;
}

bool vvPerformanceTest::getVerbose() const
{
  return _verbose;
}

float vvPerformanceTest::getQuality() const
{
  return _quality;
}

vvVector3 vvPerformanceTest::getBrickDims() const
{
  return _brickDims;
}

vvTexRend::GeometryType vvPerformanceTest::getGeomType() const
{
  return _geomType;
}

vvTexRend::VoxelType vvPerformanceTest::getVoxelType() const
{
  return _voxelType;
}

int vvPerformanceTest::getFrames() const
{
  return _frames;
}

vvPerformanceTest::TestAnimation vvPerformanceTest::getTestAnimation() const
{
  return _testAnimation;
}

vvObjView::ProjectionType vvPerformanceTest::getProjectionType() const
{
  return _projectionType;
}

vvTestResult* vvPerformanceTest::getTestResult() const
{
  return _testResult;
}

vvTestSuite::vvTestSuite(const char* pathToFile)
    : _pathToFile(pathToFile)
{
  _initialized = false;

  struct stat info;

  if (stat(pathToFile, &info) == 0)
  {
      init();
  }
}

vvTestSuite::~vvTestSuite()
{

}

bool vvTestSuite::isInitialized() const
{
  return _initialized;
}

std::vector<vvPerformanceTest*> vvTestSuite::getTests() const
{
  return _tests;
}

void vvTestSuite::init()
{
  _tests.clear();
  initColumnHeaders();

  FILE* handle = fopen(_pathToFile, "r");

  if (handle)
  {
    int lineCount = 0;
    char line[MAX_LINE_LEN];
    while (fgets(line, MAX_LINE_LEN, handle))
    {
      // One test per line (except the first one, which contains the header mapping).
      vvPerformanceTest* test = NULL;
      bool testSaved = false;
      int itemCount = 0;
      char* item;
      item = strtok(line, ",");
      char* tmp = getStripped(item);
      if (lineCount == 0)
      {
        initHeader(tmp, itemCount);
      }
      else
      {
        test = new vvPerformanceTest();
        initValue(test, tmp, itemCount);
      }
      ++itemCount;
      delete[] tmp;

      while ((item = strtok(NULL, ",")))
      {
        tmp = getStripped(item);

        if (lineCount == 0)
        {
          initHeader(tmp, itemCount);
        }
        else
        {
          initValue(test, tmp, itemCount);
        }
        ++itemCount;
        delete[] tmp;

        if ((test != NULL) && (itemCount == NUM_COL_HEADERS))
        {
          // Thus and so that the ids are more legible, they are 1-based.
          test->setId(lineCount);
          _tests.push_back(test);
          testSaved = true;
        }
      }

      if ((lineCount > 0) && (!testSaved))
      {
        test->setId(lineCount);
        _tests.push_back(test);
      }
      ++lineCount;
    }

    fclose(handle);
    _initialized = true;
  }
  else
  {
    _initialized = false;
  }
}

void vvTestSuite::initColumnHeaders()
{
  _columnHeaders[0] = "BRICKSIZE_X";    _headerPos[0] = 0;
  _columnHeaders[1] = "BRICKSIZE_Y";    _headerPos[1] = 1;
  _columnHeaders[2] = "BRICKSIZE_Z";    _headerPos[2] = 2;
  _columnHeaders[3] = "ITERATIONS";     _headerPos[3] = 3;
  _columnHeaders[4] = "QUALITY";        _headerPos[4] = 4;
  _columnHeaders[5] = "GEOMTYPE";       _headerPos[5] = 5;
  _columnHeaders[6] = "VOXELTYPE";      _headerPos[6] = 6;
  _columnHeaders[7] = "FRAMES";         _headerPos[7] = 7;
  _columnHeaders[8] = "TESTANIMATION";  _headerPos[8] = 8;
  _columnHeaders[9] = "PROJECTIONTYPE"; _headerPos[9] = 9;
}

void vvTestSuite::initHeader(char* str, const int col)
{
  toUpper(str);
  if (isHeader(str))
  {
    setHeaderPos(str, col);
  }
}

void vvTestSuite::initValue(vvPerformanceTest* test, char* str, const int col)
{
  toUpper(str);

  const char* headerName = getHeaderName(col);

  if (strcmp(headerName, "BRICKSIZE_X") == 0)
  {
    test->setBrickDimX(atoi(str));
  }
  else if (strcmp(headerName, "BRICKSIZE_Y") == 0)
  {
    test->setBrickDimY(atoi(str));
  }
  else if (strcmp(headerName, "BRICKSIZE_Z") == 0)
  {
    test->setBrickDimZ(atoi(str));
  }
  else if (strcmp(headerName, "ITERATIONS") == 0)
  {
    test->setIterations(atoi(str));
  }
  else if (strcmp(headerName, "QUALITY") == 0)
  {
    test->setQuality(atof(str));
  }
  else if (strcmp(headerName, "GEOMTYPE") == 0)
  {
    if (strcmp(str, "VV_AUTO") == 0)
    {
      test->setGeomType(vvTexRend::VV_AUTO);
    }
    else if (strcmp(str, "VV_SLICES") == 0)
    {
      test->setGeomType(vvTexRend::VV_SLICES);
    }
    else if (strcmp(str, "VV_CUBIC2D") == 0)
    {
      test->setGeomType(vvTexRend::VV_CUBIC2D);
    }
    else if (strcmp(str, "VV_VIEWPORT") == 0)
    {
      test->setGeomType(vvTexRend::VV_VIEWPORT);
    }
    else if (strcmp(str, "VV_BRICKS") == 0)
    {
      test->setGeomType(vvTexRend::VV_BRICKS);
    }
    else if (strcmp(str, "VV_SPHERICAL") == 0)
    {
      test->setGeomType(vvTexRend::VV_SPHERICAL);
    }
  }
  else if (strcmp(headerName, "VOXELTYPE") == 0)
  {
    if (strcmp(str, "VV_BEST") == 0)
    {
      test->setVoxelType(vvTexRend::VV_BEST);
    }
    else if (strcmp(str, "VV_RGBA") == 0)
    {
      test->setVoxelType(vvTexRend::VV_RGBA);
    }
    else if (strcmp(str, "VV_SGI_LUT") == 0)
    {
      test->setVoxelType(vvTexRend::VV_SGI_LUT);
    }
    else if (strcmp(str, "VV_PAL_TEX") == 0)
    {
      test->setVoxelType(vvTexRend::VV_PAL_TEX);
    }
    else if (strcmp(str, "VV_TEX_SHD") == 0)
    {
      test->setVoxelType(vvTexRend::VV_TEX_SHD);
    }
    else if (strcmp(str, "VV_PIX_SHD") == 0)
    {
      test->setVoxelType(vvTexRend::VV_PIX_SHD);
    }
    else if (strcmp(str, "VV_FRG_PRG") == 0)
    {
      test->setVoxelType(vvTexRend::VV_FRG_PRG);
    }
  }
  else if (strcmp(headerName, "FRAMES") == 0)
  {
    test->setFrames(atoi(str));
  }
  else if (strcmp(headerName, "TESTANIMATION") == 0)
  {
    if (strcmp(str, "VV_ROT_X") == 0)
    {
      test->setTestAnimation(vvPerformanceTest::VV_ROT_X);
    }
    else if (strcmp(str, "VV_ROT_Y") == 0)
    {
      test->setTestAnimation(vvPerformanceTest::VV_ROT_Y);
    }
    else if (strcmp(str, "VV_ROT_Z") == 0)
    {
      test->setTestAnimation(vvPerformanceTest::VV_ROT_Z);
    }
    else if (strcmp(str, "VV_ROT_RAND") == 0)
    {
      test->setTestAnimation(vvPerformanceTest::VV_ROT_RAND);
    }
  }
  else if (strcmp(headerName, "PROJECTIONTYPE") == 0)
  {
    if (strcmp(str, "ORTHO") == 0)
    {
      test->setProjectionType(vvObjView::ORTHO);
    }
    else if (strcmp(str, "PERSPECTIVE") == 0)
    {
      test->setProjectionType(vvObjView::PERSPECTIVE);
    }
  }
}

char* vvTestSuite::getStripped(const char* item)
{
  size_t len = strlen(item);
  if ((len >= 2) && (item[0] == '\"')
    && ((item[len - 1] == '\"') || (item[len - 1] == '\n')))
  {
    char* result = new char[len - 1];
    size_t i;
    size_t margin = (item[len - 1] == '\n') ? 3 : 2;
    for (i = 0; i < len - margin; ++i)
    {
      result[i] = item[i + 1];
    }
    result[i] = '\0';
    return result;
  }
  else
  {
    // Simply copy the string.
    char* result = new char[len + 1];
    size_t i;
    for (i = 0; i < len; ++i)
    {
      result[i] = item[i];
    }
    result[i] = '\0';
    return result;
  }
}

void vvTestSuite::toUpper(char* str)
{
  size_t len = strlen(str);

  for (size_t i = 0; i < len; ++i)
  {
    if ((str[i] >= 97) && (str[i] <= 122))
    {
      str[i] -= 32;
    }
  }
}

bool vvTestSuite::isHeader(const char* str)
{
  for (int i = 0; i < NUM_COL_HEADERS; ++i)
  {
    if ((strcmp(str, _columnHeaders[i])) == 0)
    {
      return true;
    }
  }
  return false;
}

void vvTestSuite::setHeaderPos(const char* header, const int pos)
{
  for (int i = 0; i < NUM_COL_HEADERS; ++i)
  {
    if (strcmp(header, _columnHeaders[i]) == 0)
    {
      _headerPos[i] = pos;
      break;
    }
  }
}

int vvTestSuite::getHeaderPos(const char* header)
{
  int result = -1;
  for (int i = 0; i < NUM_COL_HEADERS; ++i)
  {
    if (strcmp(header, _columnHeaders[i]) == 0)
    {
      result = _headerPos[i];
      break;
    }
  }
  return result;
}

const char* vvTestSuite::getHeaderName(const int pos)
{
  for (int i = 0; i < NUM_COL_HEADERS; ++i)
  {
    if (_headerPos[i] == pos)
    {
      return _columnHeaders[i];
    }
  }
  return NULL;
}
