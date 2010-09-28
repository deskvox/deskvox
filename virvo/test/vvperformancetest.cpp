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
#include "../src/vvgltools.h"
#include "../src/vvtoolshed.h"
#include "../src/vvvecmath.h"
#include "../src/vvvirvo.h"

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

void vvTestResult::setModelViewMatrices(const std::vector<vvMatrix> modelViewMatrices)
{
  _modelViewMatrices = modelViewMatrices;
}

std::vector<float> vvTestResult::getDiffTimes() const
{
  return _diffTimes;
}

std::vector<vvMatrix> vvTestResult::getModelViewMatrices() const
{
  return _modelViewMatrices;
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
  _outputType = VV_DETAILED;
  _datasetName = "";
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
  _testResult->calc();
  if ((_outputType == VV_SUMMARY) || (_outputType == VV_DETAILED))
  {
    // Text file with summary.
    char* summaryFile = new char[80];
    time_t now = time(NULL);
    struct tm  *ts;

    ts = localtime(&now);
    strftime(summaryFile, 80, "%Y-%m-%d_%H:%M:%S_%Z_summary.txt", ts);

    FILE* handle = fopen(summaryFile, "w");

    if (handle != NULL)
    {
      vvGLTools::GLInfo glInfo = vvGLTools::getGLInfo();
      char* dateStr = new char[80];
      strftime(dateStr, 80, "%Y-%m-%d, %H:%M:%S %Z", ts);
      fprintf(handle, "************************* Summary test %i *************************\n", _id);
      fprintf(handle, "Test performed at:....................%s\n", dateStr);
      fprintf(handle, "Virvo version:........................%s.%s\n",
              virvo::getVersionMajor(), virvo::getReleaseCounter());
      fprintf(handle, "Svn revision:.........................%s\n", virvo::getSvnRevision());
      fprintf(handle, "OpenGL vendor string:.................%s\n", glInfo.vendor);
      fprintf(handle, "OpenGL renderer string:...............%s\n", glInfo.renderer);
      fprintf(handle, "OpenGL version string:................%s\n", glInfo.version);
      fprintf(handle, "Total profiling time:.................%f\n", _testResult->getTotalTime());
      fprintf(handle, "Average time per frame:...............%f\n", _testResult->getAvgTime());
      fprintf(handle, "Variance:.............................%f\n", _testResult->getVariance());
      fprintf(handle, "Max rendering time:...................%f\n", _testResult->getMaxTime());
      fprintf(handle, "Min rendering time:...................%f\n", _testResult->getMinTime());
      fclose(handle);
    }

    if (_outputType == VV_DETAILED)
    {
      // Csv file simply with the diff times.
      char* csvFile = new char[80];
      strftime(csvFile, 80, "%Y-%m-%d_%H:%M:%S_%Z_times.csv", ts);

      handle = fopen(csvFile, "w");

      if (handle != NULL)
      {
        std::vector<float> times = _testResult->getDiffTimes();
        std::vector<vvMatrix> matrices = _testResult->getModelViewMatrices();
        std::vector<float>::const_iterator it;

        fprintf(handle, "\"TIME\",\"MODELVIEW_MATRIX\"\n");
        int i = 0;
        for (it = times.begin(); it != times.end(); ++it)
        {
          fprintf(handle, "[%f],", *it);
          fprintf(handle, "[%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f]",
                 matrices[i].e[0][0], matrices[i].e[0][1], matrices[i].e[0][2], matrices[i].e[0][3],
                 matrices[i].e[1][0], matrices[i].e[1][1], matrices[i].e[1][2], matrices[i].e[1][3],
                 matrices[i].e[2][0], matrices[i].e[2][1], matrices[i].e[2][2], matrices[i].e[2][3],
                 matrices[i].e[3][0], matrices[i].e[3][1], matrices[i].e[3][2], matrices[i].e[3][3]);
          fprintf(handle, "\n");
          ++i;
        }
        fclose(handle);
      }
    }
  }
  else if (_outputType == VV_BRICKSIZES)
  {
    // Csv file with avg time for  the given brick size.
    const int HOST_NAME_LEN = 80;
    char  localHost[HOST_NAME_LEN];
  #ifdef _WIN32
    strcpy(localHost, "n/a");
  #else
    if (gethostname(localHost, HOST_NAME_LEN-1))
    {
      strcpy(localHost, "n/a");
    }
  #endif

    char brickFile[HOST_NAME_LEN + 4];
    sprintf(brickFile, "%s.csv", localHost);
    FILE* handle;
    const bool fileExists = (handle = fopen(brickFile, "r"));
    if (fileExists)
    {
      fclose(handle);
    }
    handle = fopen(brickFile, "a+");

    // Write header if file didn't exist until now.
    if (!fileExists)
    {
      fprintf(handle, "HOSTNAME,DATASET_NAME,BRICKSIZE_X,BRICKSIZE_Y,BRICKSIZE_Z,AVG_TIME\n");
    }

    // Append timing result.
    fprintf(handle, "%s,%s,%i,%i,%i,%f\n", localHost, _datasetName,
            static_cast<int>(_brickDims[0]),
            static_cast<int>(_brickDims[1]),
            static_cast<int>(_brickDims[2]),
            _testResult->getAvgTime());
    fclose(handle);
  }
#endif
}

void vvPerformanceTest::setId(const int id)
{
  _id = id;
}

void vvPerformanceTest::setOutputType(const OutputType outputType)
{
  _outputType = outputType;
}

void vvPerformanceTest::setDatasetName(const char* datasetName)
{
  _datasetName = datasetName;
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

vvPerformanceTest::OutputType vvPerformanceTest::getOutputType() const
{
  return _outputType;
}

const char* vvPerformanceTest::getDatasetName() const
{
  return _datasetName;
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
      vvPerformanceTest* previousTest = NULL;
      if (!_tests.empty())
      {
        previousTest = _tests.back();
      }
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
        initValue(test, tmp, itemCount, previousTest);
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
          initValue(test, tmp, itemCount, previousTest);
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
  _columnHeaders[10] = "OUTPUTTYPE";    _headerPos[10] = 10;
}

void vvTestSuite::initHeader(char* str, const int col)
{
  toUpper(str);
  if (isHeader(str))
  {
    setHeaderPos(str, col);
  }
}

void vvTestSuite::initValue(vvPerformanceTest* test, char* str, const char* headerName)
{
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
    else if (strcmp(str, "VV_GLSL_SHD") == 0)
    {
      test->setVoxelType(vvTexRend::VV_GLSL_SHD);
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
  else if (strcmp(headerName, "OUTPUTTYPE") == 0)
  {
    if (strcmp(str, "VV_BRICKSIZES") == 0)
    {
      test->setOutputType(vvPerformanceTest::VV_BRICKSIZES);
    }
    else if (strcmp(str, "VV_DETAILED") == 0)
    {
      test->setOutputType(vvPerformanceTest::VV_DETAILED);
    }
    else if (strcmp(str, "VV_NONE") == 0)
    {
      test->setOutputType(vvPerformanceTest::VV_NONE);
    }
    else if (strcmp(str, "VV_SUMMARY") == 0)
    {
      test->setOutputType(vvPerformanceTest::VV_SUMMARY);
    }
  }
}

void vvTestSuite::initValue(vvPerformanceTest* test, char* str, const int col,
                            vvPerformanceTest* previousTest)
{
  toUpper(str);

  const char* headerName = getHeaderName(col);

  // TODO: fix the \n hack... .
  if ((strcmp(str, "*") == 0) || (strcmp(str, "*\n") == 0))
  {
    // * means: take the value from the previous text.
    initFromPreviousValue(test, headerName, previousTest);
  }
  else
  {
    initValue(test, str, headerName);
  }
}

void vvTestSuite::initFromPreviousValue(vvPerformanceTest* test, const char* headerName,
                                        vvPerformanceTest* previousTest)
{
  char* str = new char[256];
  if (strcmp(headerName, "BRICKSIZE_X") == 0)
  {
    sprintf(str, "%i", (int)previousTest->getBrickDims()[0]);
  }
  else if (strcmp(headerName, "BRICKSIZE_Y") == 0)
  {
    sprintf(str, "%i", (int)previousTest->getBrickDims()[1]);
  }
  else if (strcmp(headerName, "BRICKSIZE_Z") == 0)
  {
    sprintf(str, "%i", (int)previousTest->getBrickDims()[2]);
  }
  else if (strcmp(headerName, "ITERATIONS") == 0)
  {
    sprintf(str, "%i", previousTest->getIterations());
  }
  else if (strcmp(headerName, "QUALITY") == 0)
  {
    sprintf(str, "%f", previousTest->getQuality());
  }
  else if (strcmp(headerName, "GEOMTYPE") == 0)
  {
    switch (previousTest->getGeomType())
    {
    case vvTexRend::VV_AUTO:
      sprintf(str, "%s", "VV_AUTO");
      break;
    case vvTexRend::VV_SLICES:
      sprintf(str, "%s", "VV_SLICES");
      break;
    case vvTexRend::VV_CUBIC2D:
      sprintf(str, "%s", "VV_CUBIC2D");
      break;
    case vvTexRend::VV_VIEWPORT:
      sprintf(str, "%s", "VV_VIEWPORT");
      break;
    case vvTexRend::VV_BRICKS:
      sprintf(str, "%s", "VV_BRICKS");
      break;
    case vvTexRend::VV_SPHERICAL:
      sprintf(str, "%s", "VV_SPHERICAL");
      break;
    default:
      sprintf(str, "%s", "VV_AUTO");
      break;
    }
  }
  else if (strcmp(headerName, "VOXELTYPE") == 0)
  {
    switch (previousTest->getVoxelType())
    {
    case vvTexRend::VV_BEST:
      sprintf(str, "%s", "VV_BEST");
      break;
    case vvTexRend::VV_RGBA:
      sprintf(str, "%s", "VV_RGBA");
      break;
    case vvTexRend::VV_SGI_LUT:
      sprintf(str, "%s", "VV_SGI_LUT");
      break;
    case vvTexRend::VV_PAL_TEX:
      sprintf(str, "%s", "VV_PAL_TEX");
      break;
    case vvTexRend::VV_TEX_SHD:
      sprintf(str, "%s", "VV_TEX_SHD");
      break;
    case vvTexRend::VV_PIX_SHD:
      sprintf(str, "%s", "VV_PIX_SHD");
      break;
    case vvTexRend::VV_FRG_PRG:
      sprintf(str, "%s", "VV_FRG_PRG");
      break;
    case vvTexRend::VV_GLSL_SHD:
      sprintf(str, "%s", "VV_GLSL_SHD");
      break;
    default:
      sprintf(str, "%s", "VV_BEST");
      break;
    }
  }
  else if (strcmp(headerName, "FRAMES") == 0)
  {
    sprintf(str, "%i", previousTest->getFrames());
  }
  else if (strcmp(headerName, "TESTANIMATION") == 0)
  {
    switch (previousTest->getTestAnimation())
    {
    case vvPerformanceTest::VV_ROT_X:
      sprintf(str, "%s", "VV_ROT_X");
      break;
    case vvPerformanceTest::VV_ROT_Y:
      sprintf(str, "%s", "VV_ROT_Y");
      break;
    case vvPerformanceTest::VV_ROT_Z:
      sprintf(str, "%s", "VV_ROT_Z");
      break;
    case vvPerformanceTest::VV_ROT_RAND:
      sprintf(str, "%s", "VV_ROT_RAND");
      break;
    default:
      sprintf(str, "%s", "VV_ROT_Y");
      break;
    }
  }
  else if (strcmp(headerName, "PROJECTIONTYPE") == 0)
  {
    switch (previousTest->getProjectionType())
    {
    case vvObjView::FRUSTUM:
      sprintf(str, "%s", "FRUSTUM");
      break;
    case vvObjView::ORTHO:
      sprintf(str, "%s", "ORTHOG");
      break;
    case vvObjView::PERSPECTIVE:
      sprintf(str, "%s", "PERSPECTIVE");
      break;
    default:
      sprintf(str, "%s", "PERSPECTIVE");
      break;
    }
  }
  else if (strcmp(headerName, "OUTPUTTYPE") == 0)
  {
    switch (previousTest->getOutputType())
    {
    case vvPerformanceTest::VV_BRICKSIZES:
      sprintf(str, "%s", "VV_BRICKSIZES");
      break;
    case vvPerformanceTest::VV_DETAILED:
      sprintf(str, "%s", "VV_DETAILED");
      break;
    case vvPerformanceTest::VV_NONE:
      sprintf(str, "%s", "VV_NONE");
      break;
    case vvPerformanceTest::VV_SUMMARY:
      sprintf(str, "%s", "VV_SUMMARY");
      break;
    default:
      sprintf(str, "%s", "VV_DETAILED");
      break;
    }
  }
  initValue(test, str, headerName);
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
