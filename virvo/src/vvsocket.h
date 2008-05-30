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

#ifndef _VVSOCKET_H_
#define _VVSOCKET_H_

#include <iostream>

#ifdef _WIN32
#ifndef COVISE
# include <winsock2.h>
#endif
# include <windows.h>
# include <time.h>
#else
# include <netdb.h>
# include <unistd.h>
# include <arpa/inet.h>
# include <netinet/in.h>
# include <netinet/tcp.h>
# include <sys/time.h>
# include <sys/errno.h>
# include <sys/param.h>
# include <sys/ioctl.h>
# include <sys/socket.h>
# include <sys/wait.h>
# include <errno.h>
#endif
# include <string.h>
# include <signal.h>
# include <assert.h>
# include <stdlib.h>
#ifdef __sun
#include <sys/filio.h>
#endif

#include "vvexport.h"

//============================================================================
// Type Declarations
//============================================================================

#ifdef _WIN32
/* code copied from Python's pyconfig.h,
 * to avoid different definition of  ssize_t */
#ifdef _WIN64
typedef __int64 ssize_t;
#else
typedef _W64 int ssize_t;
#endif
/* end copy */
#endif

typedef void Sigfunc(int);
typedef unsigned char   uchar;                    ///< abbreviation for unsigned char
typedef unsigned short  ushort;                   ///< abbreviation for unsigned short
typedef unsigned int    uint;                     ///< abbreviation for unsigned int
typedef unsigned long   ulong;                    ///< abbreviation for unsigned long

#define VV_TRACE( DBG_LVL, ACT_LVL, STRING) \
  { \
    if (ACT_LVL >= DBG_LVL) \
      cerr<<STRING<<endl; \
  }

#define VV_ERRNO( DBG_LVL, ACT_LVL, STRING) \
  { \
    if (ACT_LVL >= DBG_LVL) \
      printErrorMessage(STRING); \
  }

//----------------------------------------------------------------------------
/** This class provides basic socket functionality. It is used for TCP and UDP
    sockets. For example code see documentation about vvSocket  <BR>
*/
class VIRVOEXPORT vvSocket
{

  public:
    enum ErrorType                                /// Error Codes
    {
      VV_OK,                                      ///< no error
      VV_TIMEOUT_ERROR,
      VV_SOCK_ERROR,
      VV_WRITE_ERROR,
      VV_READ_ERROR,
      VV_ACCEPT_ERROR,
      VV_CONNECT_ERROR,
      VV_HOST_ERROR,
      VV_RETRY,
      VV_ALLOC_ERROR,                             ///< allocation error: not enough memory
      VV_CREATE_ERROR,                            ///< socket could not be opened
      VV_HEADER_ERROR,                            ///< invalid header received
      VV_DATA_ERROR                               ///< volume data format error: e.g., too many voxels received
    };

    enum SocketType
    {
      VV_TCP,
      VV_UDP
    };

    enum EndianType                               /// endianness
    {
      VV_LITTLE_END,                              ///< little endian: low-order byte is stored first
      VV_BIG_END                                  ///< big endian: hight-order byte is stored first
    };

    vvSocket( int, SocketType socktype = VV_TCP);
    vvSocket(int, const char*, SocketType socktype = VV_TCP, int clminport = 0 , int clmaxport = 0);
    virtual ~vvSocket();

    ErrorType init();
    ErrorType read_data(uchar*, size_t);
    ErrorType write_data(const uchar*, size_t);
    void set_debuglevel(int);
    void set_sock_buffsize(int);
    void set_timer(float, float);
    int is_data_waiting();
    ErrorType read_string(char* , int);
    ErrorType write_string(const char*);
    int get_sockfd();
    int get_recv_buffsize();
    int get_send_buffsize();
    uchar     read8();
    ErrorType write8(uchar);
    ushort    read16(EndianType = VV_BIG_END);
    ErrorType write16(ushort, EndianType = VV_BIG_END);
    uint      read32(EndianType = VV_BIG_END);
    ErrorType write32(uint, EndianType = VV_BIG_END);
    float     readFloat(EndianType = VV_BIG_END);
    ErrorType writeFloat(float, EndianType = VV_BIG_END);
    int no_nagle();
    int set_linger(int);
    int get_MTU();

  protected:
    enum {NUM_TIMERS = 2};
    struct sockaddr_in host_addr;
    struct hostent *host;
    int sockfd, port;
    const char* hostname;
    SocketType socktype;
    int cl_min_port, cl_max_port;
    int debuglevel, sock_buffsize;
    float transfer_timer;
    int connect_timer;
    int recv_buffsize, send_buffsize;
    int max_send_size;

#if !defined(__linux__) && !defined(LINUX) && !(defined(__APPLE__) && defined(__GNUC__) && GNUC__ < 4)
#define socklen_t int
#endif
    socklen_t host_addrlen, bufflen;

#ifdef _WIN32
    clock_t t_start[NUM_TIMERS];
#else
    timeval t_start[NUM_TIMERS];
#endif

    Sigfunc *signal(int, Sigfunc *);
    Sigfunc *Signal(int, Sigfunc *);
    static void nonameserver(int );
    int writeable_timeo();
    int readable_timeo();
    ErrorType read_timeo(uchar*, size_t);
    ErrorType write_timeo(const uchar*, size_t);
    ErrorType read_nontimeo(uchar*, size_t);
    ErrorType write_nontimeo(const uchar*, size_t);
    void printErrorMessage(const char* = NULL);
    bool do_a_retry();
    void startTime(int);
    float getTime(int);
    ErrorType init_server_tcp();
    ErrorType init_client_tcp();
    ErrorType init_server_udp();
    ErrorType init_client_udp();
    ErrorType get_client_addr();
    ErrorType recvfrom_timeo();
    ErrorType recvfrom_nontimeo();
    ssize_t readn_tcp(char*, size_t);
    ssize_t writen_tcp(const char*, size_t);
    ssize_t readn_udp(char*, size_t);
    ssize_t writen_udp(const char*, size_t);
    static void interrupter(int );
    ErrorType accept_timeo();
    ErrorType connect_timeo();
    ErrorType accept_nontimeo();
    ErrorType connect_nontimeo();
    int measure_BDP_server();
    int measure_BDP_client();
    float RTT_client(int);
    int RTT_server(int);
    int checkMSS_MTU(int, int);
    EndianType getEndianness();
};

//----------------------------------------------------------------------------
/***TCP Sockets***.  <BR>

 Features:
    - timeouts for connection establishment and data transfer
    - socket buffer sizes can be set be user
    - automatic bandwidth delay product discovery to set the socket buffers to the
      optimal values. Not supported under Windows and when VV_BDP Flag is not set.
      For automatic banwidth delay product discovery the socket buffer size has
      to be set to 0. Optimized for networks with more than 10 Mbits/sec. Please
      don't use if you have a lower speed (would take awhile).
    - Nagle algorithm can be disabled
- Linger time can be set<BR>

Default values:
- debuglevel=1 (0 lowest level and no messages, 1 only error messages, 2 some more messages, 3 highest level)
- socket buffer size= system default
- no timeouts<BR>

Here is an example code fragment to generate a TCP server which sends 10 bytes
and a TCP-client which reads 10 bytes.<BR>
<PRE>

TCP-Server:

// Create a new tcp socket class instance which shall listen on port 17171:
vvSocket* sock = new vvSocket(17171, vvSocket::VV_TCP);

// Parameters must be set before the init() call !!
// e.g. debuglevel=3, socket buffer size= 65535 byte, \ 
// timer for accept=3 sec., timer for write=1.5 sec.
sock->set_timer(3.0f, 1.5f);
sock->set_sock_buffsize(65535);
sock->set_debuglevel(3);

// Initialize the socket with the parameters and wait for a server
if (sock->init() != vvSocket::VV_OK)
{
delete sock;
return -1;
}

// Send 10 bytes of data with write_data()
uchar buffer[10];
if (sock->write_data(&buffer, 10) != vvSocket::VV_OK)
{
delete sock;
return -1;
}

// Delete the socket object
delete sock;

TCP-Client:

// Create a new tcp socket class instance which shall connect to a server
// with name buxdehude on port 17171. The outgoing port shall be in the
// range between 31000 and 32000:
char* servername = "buxdehude";
vvSocket* sock = new vvSocket(17171, servername, vvSocket::VV_TCP, 31000, 32000);

// Parameters must be set before the init() call !!
// e.g. debuglevel=3, socket buffer size= 65535 byte, \ 
// timer for connect=3 sec., timer for read=1.5 sec.
sock->set_timer(3.0f, 1.5f);
sock->set_sock_buffsize(65535);
sock->set_debuglevel(3);

// Initialize the socket with the parameters and connect to the server.
if (sock->init() != vvSocket::VV_OK)
{
delete sock;
return -1;
}

// Get 10 bytes of data with read_data()
uchar buffer[10];
if (sock->read_data(&buffer, 10) != vvSocket::VV_OK)
{
delete sock;
return -1;
}

// Delete the socket object
delete sock; </PRE>
@author Michael Poehnl
*/

//----------------------------------------------------------------------------
/***For UDP Sockets***  <BR>

 features:
    - timeouts for "connection establishment"(here connected UDP sockets
    are used) and data transfer
    - socket buffer sizes can be set be user<BR>

 default values:
   - debuglevel=1 (0 lowest level, 3 highest level)
   - socket buffer size= system default
   - no timeouts<BR>

Here is an example code fragment to generate a UDP server which sends 10 bytes
and a UDP client which reads 10 bytes.<BR>
<PRE>

UDP-Server:

// Create a new UDP socket class instance which shall listen on port 17171:
vvSocket* sock = new vvSocket(17171, vvSocket::VV_UDP);

// Parameters must be set before the init() call !!
// e.g. debuglevel=3, socket buffer size= 65535 byte, \ 
// timer for connect=3 sec., timer for write=1.5 sec.
sock->set_timer(3.0f, 1.5f);
sock->set_sock_buffsize(65535);
sock->set_debuglevel(3);

// Initialize the socket with the parameters and wait for a server
if (sock->init() != vvSocket::VV_OK)
{
delete sock;
return -1;
}

// Send 10 bytes of data with write_data()
uchar buffer[10];
if (sock->write_data(&buffer, 10) != vvSocket::VV_OK)
{
delete sock;
return -1;
}

// Delete the socket object
delete sock;

UDP-Client:

// Create a new UDP socket class instance which shall connect to a server
// with name buxdehude on port 17171. The outgoing port shall be in the
// range between 31000 and 32000:
char* servername = "buxdehude";
vvSocket* sock = new vvSocket(17171, servername, vvSocket::VV_UDP, 31000, 32000);

// Parameters must be set before the init() call !!
// e.g. debuglevel=3, socket buffer size= 65535 byte, \ 
// timer for connect=3 sec., timer for read=1.5 sec.
sock->set_timer(3.0f, 1.5f);
sock->set_sock_buffsize(65535);
sock->set_debuglevel(3);

// Initialize the socket with the parameters and connect to the server.
if (sock->init() != vvSocket::VV_OK)
{
delete sock;
return -1;
}

// Get 10 bytes of data with read_data()
uchar buffer[10];
if (sock->read_data(&buffer, 10) != vvSocket::VV_OK)
{
delete sock;
return -1;
}

// Delete the socket object
delete sock; </PRE>
@author Michael Poehnl
*/
#endif
