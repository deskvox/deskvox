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

#ifdef _WIN32
#include <windows.h>                              // required for QueryPerformance API
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvstopwatch.h"

//#define VV_STANDALONE      // uncomment for demonstration

//----------------------------------------------------------------------------
/// Constructor. Initializes time variable with zero.
vvStopwatch::vvStopwatch()
{
#ifdef _WIN32
  baseTime = 0;
  baseTimeQP.QuadPart = 0;
  useQueryPerformance = (QueryPerformanceFrequency(&freq)) ? true : false;
#else
  baseTime.tv_sec  = 0;
  baseTime.tv_usec = 0;
#endif
  lastTime = 0.0f;
}

//----------------------------------------------------------------------------
/// Start or restart measurement but don't reset counter.
void vvStopwatch::start()
{
#ifdef _WIN32

  if (useQueryPerformance) QueryPerformanceCounter(&baseTimeQP);
  else baseTime = clock();

#elif defined(__linux__) || defined(LINUX) || defined(__APPLE__)

  struct timezone tz;
  gettimeofday(&baseTime, &tz);

#else

  void* v = NULL;
  gettimeofday(&baseTime, v);
#endif

  lastTime = 0.0f;
}

//----------------------------------------------------------------------------
/// Return the time passed since the last start command [seconds].
float vvStopwatch::getTime()
{
  float dt;                                       // measured time difference [seconds]

#ifdef _WIN32

  if (useQueryPerformance)
  {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    dt = float((float(now.QuadPart) - float(baseTimeQP.QuadPart)) / float(freq.QuadPart));
  }
  else
  {
    clock_t now = clock();
    dt = float(now - baseTime) / float(CLOCKS_PER_SEC);
  }

#else

#if defined(__linux__) || defined(LINUX) || defined(__APPLE__)
  struct timezone dummy;
#else
  void* dummy = NULL;
#endif
  timeval now;                                    // current system time

  gettimeofday(&now, &dummy);
  time_t sec  = now.tv_sec  - baseTime.tv_sec;
  long   usec = now.tv_usec - baseTime.tv_usec;
  dt   = (float)sec + (float)usec / 1000000.0f;
#endif

  lastTime = dt;
  return dt;
}

//----------------------------------------------------------------------------
/// Return the time passed since the last getTime or getDiff command [seconds].
float vvStopwatch::getDiff()
{
  float last = lastTime;
  return getTime() - last;
}

//============================================================================
// Functions for STANDALONE mode
//============================================================================

#ifdef VV_STANDALONE

#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
  vvStopwatch* watch;
  char input[128];

  watch = new vvStopwatch();

  cerr << "Input something to start stopwatch: " << endl;
  cin >> input;
  watch->start();

  cerr << "Input something to stop watch: " << endl;
  cin >> input;
  watch->stop();
  cerr << "Current time: " << watch->getTime() << endl;

  cerr << "Input something to continue taking time: " << endl;
  cin >> input;
  watch->start();

  cerr << "Input something to stop watch: " << endl;
  cin >> input;
  watch->stop();
  cerr << "Total time: " << watch->getTime() << endl;

  delete watch;

  return 0;
}
#endif

//============================================================================
// End of File
//============================================================================
