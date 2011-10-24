// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
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
#include <cstring>
#include <cstdlib>
#include <cstdio>

#include "vvmulticast.h"
#include "vvsocket.h"

using namespace std;

int main(int argc, char** argv)
{
  // don't forget arguments
  if(argc < 2)
  {
    cout << "Start with -sender or -receiver" << endl;
    return 0;
  }

  // Number of Bytes to be send/received
  uint count = 1024;
  if(NULL != argv[2] && 3 <= argc)
  {
    count = atoi(argv[2]);
  }
  else
  {
    cout << "No number of bytes given. (Default: " << count << ")" << endl;
  }

  // init random number generator
  srand(123456);

  // init data to be sent
  cout << "Prepare " << count << " Bytes of random data to be send/checked..." << flush;
  uchar* bar = new uchar[count];
  for(int i=0; i<count; i++)
  {
    uchar num = rand() % 256;
    bar[i] = num;
  }
  cout << "done!" << endl;

  // ----------------------------------------------------------------
  // Sender
  // ----------------------------------------------------------------
  if(strcmp("-sender", argv[1])== 0)
  {
    cout << "Sender-Mode" << endl;
    cout << "###########" << endl;
    cout << endl;

    // timeout
    double sendTimeout;
    if(NULL != argv[3] && 4 <= argc)
    {
      sendTimeout = atof(argv[3]);
    }
    else
    {
      cout << "No timeout given. (Default: no timeout)" << endl;
      sendTimeout = -1.0;
    }

    // init Multicaster
    vvMulticast foo = vvMulticast("224.1.2.3", 50096, vvMulticast::VV_SENDER);

    cout << "Sending " << count << " Bytes of random numbers..." << flush;
    int sendBytes = foo.write(reinterpret_cast<const unsigned char*>(bar), count, sendTimeout);
    cout << "done!" << endl;

    if(sendBytes == -1.0)
      cout << "Error occured! (No Norm found?)" << sendBytes << endl;
    else
      cout << "Successfully sent " << sendBytes << " Bytes!" << endl;

    cout << endl;

    cout << "Type something when Transfer on receiver-side done...." << endl;

    cout << endl;

    string hold;
    cin >> hold;

    delete[] bar;

    return 0;
  }

  // ----------------------------------------------------------------
  // Receiver
  // ----------------------------------------------------------------
  if(strcmp("-receiver", argv[1])== 0)
  {
    cout << "Receiver-Mode" << endl;
    cout << "#############" << endl;
    cout << endl;

    // timeout
    double receiveTimeout;
    if(NULL != argv[3] && 4 <= argc)
    {
      receiveTimeout = atof(argv[3]);
    }
    else
    {
      cout << "No timeout given. (Default: no timeout)" << endl;
      receiveTimeout = -1.0;
    }

    cout << "Waiting for incoming data..." << endl;
    vvMulticast foo = vvMulticast("224.1.2.3", 50096, vvMulticast::VV_RECEIVER);
    uchar* bartext = new uchar[count];
    int receivedBytes = foo.read(count, bartext, receiveTimeout);
    cout << "Received: " << receivedBytes << endl;
    if(0 == receivedBytes)
      cout << "Timeout reached and no data received!" << endl;
    cout << endl;

    cout << "Check data for differences...    ";
    for(int i=0; i<receivedBytes;i++)
    {
      if(bar[i] != bartext[i])
      {
        cout << "Failed: Differences found!" << endl;
        cout << bar[i] << " != " << bartext[i] << endl;
        break;
      }
      else if(i % 1024 == 0)
      {
        cout << "\r" << flush;
        cout << "Check data for differences..." << int(100 * float(i)/float(count)) << "%" << flush;
      }
    }
    cout << endl;

    delete[] bar;
    delete[] bartext;
    return 0;
  }

  cout << "Nothing done..." << endl;

  return 1;
}


/*

Build-Notes:

build with libraries: virvo, norm, Protokit

Attention!
If Protokit and Norm are build seperately, then use libProto.a instead of libProtokit.a! (both libs are build automatically)
If Protokit is build together with norm (included subdirectory) then use libProtokit.a (only this lib is build)
(Same thing for windows and .dlls)
The reason for this issue is, that norm-developers use an older/different version of Protokit and are too lazy to fix this.
Norm is generally build with the old version instead.


// g++ vvmulticasttest.cpp -I/raid/home/sdelisav/deskvox/virvo/virvo -I/raid/home/sdelisav/Desktop/norm-1.4b3/common -L /raid/home/sdelisav/Desktop/norm-1.4b3/unix/ -I /raid/home/sdelisav/deskvox/qtcreator-build/virvo/ -L /raid/home/sdelisav/deskvox/qtcreator-build/virvo/virvo/ -l virvo -l norm -L /raid/home/sdelisav/Desktop/norm-1.4b3/protolib/unix/ -l Protokit -pthread -DHAVE_NORM

*/

