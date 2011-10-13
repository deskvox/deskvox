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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#if HAVE_NORM

#include <iostream>
#include <cstring>

#include "vvmulticast.h"

using namespace std;

int main(int argc, char** argv)
{
  // don't forget arguments
  if(argc < 2)
  {
    cout << "Start with -sender text or -receiver" << endl;
    return 0;
  }

  // Sender
  if(strcmp("-sender", argv[1])== 0)
  {
    cout << "Sender-Mode" << endl;
    vvMulticast* foo = new vvMulticast("224.1.2.3", 50096, vvMulticast::VV_SENDER);

    string footext = string(argv[2]);
    footext.resize(200, ' ');

    foo->write(reinterpret_cast<const unsigned char*>(footext.c_str()), 500, 3.0);

    return 0;
  }

  // Receiver, start with -receiver
  if(strcmp("-receiver", argv[1])== 0)
  {
    cout << "Receiver-Mode" << endl;
    vvMulticast* bar = new vvMulticast("224.1.2.3", 50096, vvMulticast::VV_RECEIVER);
    unsigned char* bartext;
    bar->read(200, bartext);
    std::cout << "Received: " << reinterpret_cast<char*>(bartext) << std::endl;
    return 0;
  }

  cout << "Nothing done..." << endl;
  return 1;
}

#endif

/*

Build-Notes:

build with libraries: virvo, norm, Protokit

Attention!
If Protokit and Norm are build seperately, then use libProto.a instead of libProtokit.a! (both libs are build automatically)
If Protokit is build together with norm (included subdirectory) then use libProtokit.a (only this lib is build)
(Same thing for windows and .dll's)
The reason for this issue is, that norm-developers use an older/different version of Protokit and are too lazy to fix this.
Norm is generally build with the old version instead.


// g++ vvmulticasttest.cpp -I/raid/home/sdelisav/deskvox/virvo/virvo -I/raid/home/sdelisav/Desktop/norm-1.4b3/common -L /raid/home/sdelisav/Desktop/norm-1.4b3/unix/ -I /raid/home/sdelisav/deskvox/qtcreator-build/virvo/ -L /raid/home/sdelisav/deskvox/qtcreator-build/virvo/virvo/ -l virvo -l norm -L /raid/home/sdelisav/Desktop/norm-1.4b3/protolib/unix/ -l Protokit -pthread -DHAVE_NORM

*/

