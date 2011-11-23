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

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include <cassert>
#include <sstream>

#include "vvsocket.h"
#include "vvdebugmsg.h"

#ifndef _WIN32
#include <signal.h>
#endif

using namespace std;

//----------------------------------------------------------------------------
/** Interrupter
 */
void vvSocket::interrupter(int)
{
  return;                                         // just interrupt
}

//----------------------------------------------------------------------------
/// Initializes a TCP server
vvSocket::ErrorType vvSocket::init_server_tcp()
{
#ifdef _WIN32
  char optval=1;
#else
  int optval=1;
#endif
  ErrorType retval;

  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0 )) < 0)
  {
    vvDebugMsg::msg(1, "Error: socket()");
    return VV_SOCK_ERROR;
  }
  if (sock_buffsize == 0)
  {
#if !defined(_WIN32) && defined(VV_BDP)
    if(measure_BDP_server())
    {
      vvDebugMsg::msg(1, "Error: measure_BDP_server()");
      return VV_SOCK_ERROR;
    }
#else
    sock_buffsize = get_send_buffsize();
#endif
  }
  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval,sizeof(optval)))
  {
    vvDebugMsg::msg(1, "Error: setsockopt()");
    return VV_SOCK_ERROR;
  }
  if (sock_buffsize > 0)
  {
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF,
      (char *) &sock_buffsize, sizeof(sock_buffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF,
      (char *) &sock_buffsize, sizeof(sock_buffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
  }
  memset((char *) &host_addr, 0, sizeof(host_addr));
  host_addr.sin_family = AF_INET;
  host_addr.sin_port = htons((unsigned short)port);
  host_addr.sin_addr.s_addr = INADDR_ANY;
  host_addrlen = sizeof(host_addr);
  if  (bind(sockfd, (struct sockaddr *)&host_addr, host_addrlen))
  {
    vvDebugMsg::msg(1, "Error: bind()");
    return VV_SOCK_ERROR ;
  }

  if (listen(sockfd, 1))
  {
    vvDebugMsg::msg(1, "Error: listen()");
    return VV_SOCK_ERROR;
  }
  if (connect_timer > 0)
  {
    do
    retval=accept_timeo();
    while( retval == VV_RETRY);
  }
  else
    retval=accept_nontimeo();
  ostringstream errmsg;
  errmsg << "send_buffsize: " << get_send_buffsize() << " bytes, recv_buffsize: " << get_recv_buffsize() << " bytes";
  vvDebugMsg::msg(2, errmsg.str().c_str());
  if (retval == VV_OK)
  {
    ostringstream errmsg;
    errmsg << "Incoming connection from " << inet_ntoa(host_addr.sin_addr);
    vvDebugMsg::msg(2, errmsg.str().c_str());
  }
  return retval;
}

//----------------------------------------------------------------------------
/// Initializes a TCP client
vvSocket::ErrorType vvSocket::init_client_tcp()
{
  ErrorType retval;
  int cl_port;

#ifdef _WIN32
  if ((host= gethostbyname(hostname)) == 0)
  {
    vvDebugMsg::msg(1, "Error: gethostbyname()");
    return VV_HOST_ERROR;
  }
#else
  Sigfunc *sigfunc;

  sigfunc = Signal(SIGALRM, nonameserver);
  if (alarm(5) != 0)
    vvDebugMsg::msg(2, "init_client(): WARNING! previously set alarm was wiped out");
  if ((host= gethostbyname(hostname)) == 0)
  {
    vvDebugMsg::msg(1, "Error: gethostbyname()");
    alarm(0);
    signal(SIGALRM, sigfunc);
    return VV_HOST_ERROR;
  }
  alarm(0);
  signal(SIGALRM, sigfunc);
#endif
  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
  {
    vvDebugMsg::msg(1, "Error: socket");
    return VV_SOCK_ERROR;
  }
  if (sock_buffsize == 0)
  {
#if !defined(_WIN32) && defined(VV_BDP)
    if (measure_BDP_client())
    {
      vvDebugMsg::msg(1, "Error: measure_BDP_client()");
      return VV_SOCK_ERROR;
    }
#else
    sock_buffsize = get_send_buffsize();
#endif
  }
  if (sock_buffsize > 0)
  {
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF,
      (char *) &sock_buffsize, sizeof(sock_buffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF,
      (char *) &sock_buffsize, sizeof(sock_buffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
  }
  memset((char *) &host_addr, 0, sizeof(host_addr));
  host_addr.sin_family = AF_INET;
  host_addrlen = sizeof(host_addr);
  if (cl_min_port != 0 || cl_max_port != 0)
  {
    if (cl_min_port > cl_max_port)
    {
      vvDebugMsg::msg(1,"Wrong port range");
      return VV_SOCK_ERROR ;
    }
    host_addr.sin_addr.s_addr = INADDR_ANY;
    cl_port = cl_min_port;
    host_addr.sin_port = htons((unsigned short)cl_port);
    while (bind(sockfd, (struct sockaddr *)&host_addr, host_addrlen) && cl_port <= cl_max_port)
    {
#ifdef _WIN32
      if (WSAGetLastError() == WSAEADDRINUSE)
#else
        if (errno == EADDRINUSE)
#endif
      {
        cl_port ++;
        host_addr.sin_port = htons((unsigned short)cl_port);
      }
      else
      {
        vvDebugMsg::msg(1, "Error: bind()");
        return VV_SOCK_ERROR ;
      }
    }
    if (cl_port > cl_max_port)
    {
      vvDebugMsg::msg(1,"No port free!");
      return VV_SOCK_ERROR ;
    }
  }
  host_addr.sin_addr = *((struct in_addr *)host->h_addr);
  host_addr.sin_port = htons((unsigned short)port);
  if (connect_timer > 0)
  {
    do
    retval = connect_timeo();
    while (retval == VV_RETRY);
  }
  else
    retval = connect_nontimeo();
  ostringstream errmsg;
  errmsg << "send_buffsize: " << get_send_buffsize() << " bytes, recv_buffsize: " << get_recv_buffsize() << " bytes";
  vvDebugMsg::msg(2, errmsg.str().c_str());
  return retval;
}

//----------------------------------------------------------------------------
/**Calls accept() with a timeout. Length of timeout is connect_timer.
 */
vvSocket::ErrorType vvSocket::accept_timeo()
{
#ifdef _WIN32
  unsigned long cmd_option = 1;

  if (ioctlsocket(sockfd, FIONBIO, &cmd_option) == SOCKET_ERROR)
    return VV_ACCEPT_ERROR;
  int ar = INVALID_SOCKET;
  DWORD start = GetTickCount();
  while (ar == INVALID_SOCKET)
  {
    ar = accept(sockfd, (struct sockaddr *)&host_addr, &host_addrlen);
    if (ar == INVALID_SOCKET)
    {
      vvDebugMsg::msg(2, "Error: accept()");
      if ((GetTickCount() - start) > transfer_timer*1000)
        return VV_TIMEOUT_ERROR;
    }
  }
  sockfd = ar;
  cmd_option = 0;
  if (ioctlsocket(sockfd, FIONBIO, &cmd_option) == SOCKET_ERROR)
    return VV_ACCEPT_ERROR;

#else
  Sigfunc *sigfunc;
  int n;

  sigfunc = Signal(SIGALRM, interrupter);
  if (alarm(connect_timer) != 0)
    vvDebugMsg::msg(2, "accept_timeo(): WARNING! previously set alarm was wiped out");
  if ( (n = accept(sockfd, (struct sockaddr *)&host_addr, &host_addrlen)) < 0)
  {
    alarm(0);
    if (errno == EINTR)
    {
      if (vvDebugMsg::getDebugLevel()>1)
      {
        if (do_a_retry())
          return VV_RETRY;
      }
      signal(SIGALRM, sigfunc);
      return VV_TIMEOUT_ERROR;
    }
    else
    {
      vvDebugMsg::msg(1, "Error: accept()");
      signal(SIGALRM, sigfunc);
      return VV_ACCEPT_ERROR;
    }
  }
  alarm(0);                                       /* turn off the alarm */
  close(sockfd);
  sockfd = n;
  signal(SIGALRM, sigfunc);
#endif
  return VV_OK;
}

//----------------------------------------------------------------------------
/**Calls accept() without a timeout.
 */
vvSocket::ErrorType vvSocket::accept_nontimeo()
{
  int n;

  if ( (n = accept(sockfd, (struct sockaddr *)&host_addr, &host_addrlen)) < 0)
  {
    vvDebugMsg::msg(1, "Error: accept()");
    return VV_ACCEPT_ERROR;
  }

#ifdef _WIN32
  closesocket(sockfd);
#else
  close(sockfd);
#endif
  sockfd = n;
  return VV_OK;
}

//----------------------------------------------------------------------------
/**Calls connect() with a timeout. Length of timeout is connect_timer.
 */
vvSocket::ErrorType vvSocket::connect_timeo()
{
#ifdef _WIN32
  unsigned long cmd_option = 1;
  if (ioctlsocket(sockfd, FIONBIO, &cmd_option) == SOCKET_ERROR)
    return VV_CONNECT_ERROR;
  int cr = 1;
  int error;
  DWORD start = GetTickCount();
  while (cr != 0)
  {
    cr = connect(sockfd, (struct sockaddr*)&host_addr, host_addrlen);
    if (cr != 0)
    {
      error = WSAGetLastError();
      if (error==WSAEISCONN) cr = 0;              // this is a weird Windows specific necessity!
      if (error!=WSAEALREADY && error!=WSAEWOULDBLOCK && error!=10022)
        vvDebugMsg::msg(1, "Error: connect()");
      if ((GetTickCount() - start) > transfer_timer*1000)
        return VV_TIMEOUT_ERROR;
    }
  }

  vvDebugMsg::msg(2, "connection established");
  cmd_option = 0;
  if (ioctlsocket(sockfd, FIONBIO, &cmd_option) == SOCKET_ERROR)
    return VV_CONNECT_ERROR;

#else
  Sigfunc *sigfunc;

  sigfunc = Signal(SIGALRM, interrupter);
  if (alarm(connect_timer) != 0)
    vvDebugMsg::msg(2, "connect_timeo:WARNING! previously set alarm was wiped out");
  if (connect(sockfd, (struct sockaddr *)&host_addr, host_addrlen))
  {
    alarm(0);
    if (errno == EINTR)
    {
      if (vvDebugMsg::getDebugLevel()>1)
      {
        if (do_a_retry())
          return VV_RETRY;
      }
      signal(SIGALRM, sigfunc);
      return VV_TIMEOUT_ERROR;
    }
    else
    {
      vvDebugMsg::msg(1, "Error: connect()");
      signal(SIGALRM, sigfunc);
      return VV_CONNECT_ERROR;
    }
  }
  alarm(0);                                       /* turn off the alarm */
  signal(SIGALRM, sigfunc);                       /* restore previous signal handler */
#endif
  return VV_OK;
}

//----------------------------------------------------------------------------
/**Calls connect() without a timeout.
 */
vvSocket::ErrorType vvSocket::connect_nontimeo()
{
  if (connect(sockfd, (struct sockaddr *)&host_addr, host_addrlen))
  {
    vvDebugMsg::msg(1, "Error: connect()");
    return VV_CONNECT_ERROR;
  }
  return VV_OK;
}

//----------------------------------------------------------------------------
/**Reads data from the TCP socket.
 @param buffer  pointer to the data to write
 @param size   number of bytes to write

*/
ssize_t vvSocket::readn_tcp(char* buffer, size_t size)
{
  size_t nleft;
  ssize_t nread;

  nleft = size;
  while(nleft > 0)
  {
    nread = recv(sockfd, buffer, nleft, 0);
    if (nread < 0)
    {
#ifdef _WIN32
      if (WSAGetLastError() == WSAEINTR)
#else
        if (errno == EINTR)
#endif
          nread = 0;                              // interrupted, call read again
      else
      {
        vvDebugMsg::msg(1, "Error: recv()");
        return (ssize_t)-1;
      }
    }
    else if (nread == 0)
      break;

    nleft -= nread;
    buffer += nread;
  }
  return (size - nleft);
}

//----------------------------------------------------------------------------
/**Writes data to the TCP socket.
 @param buffer  pointer to the data to write
 @param size   number of bytes to write

*/
ssize_t vvSocket::writen_tcp(const char* buffer, size_t size)
{
  size_t nleft;
  ssize_t nwritten;

#ifndef _WIN32
  ::signal(SIGPIPE, peerunreachable);
#endif

  nleft = size;
  while(nleft > 0)
  {
    nwritten = send(sockfd, buffer, nleft, 0);
    if (nwritten < 0)
    {
#ifdef _WIN32
      if (WSAGetLastError() == WSAEINTR)
#else
        if (errno == EINTR)
#endif
          nwritten = 0;                           // interrupted, call write again
      else
      {
        vvDebugMsg::msg(1, "Error: send()");
        return (ssize_t)-1;
      }
    }

    nleft -= nwritten;
    buffer += nwritten;
  }

  return size;
}

//----------------------------------------------------------------------------
/**Server for round-trip-time measurement. Needed for bandwidth-delay-product.
 NOT SUPPORTED UNDER WINDOWS AND WHEN THE VV_BDP FLAG IS NOT SET
 @param payload   payload size in bytes for UDP packets
*/
int vvSocket::RTT_server(int payload)
{
#if !defined(_WIN32) && defined(VV_BDP)
  uchar* frame;
  ErrorType retval;

  vvSocket* usock = new vvSocket(port, VV_UDP);
  usock->set_debuglevel(debuglevel);
  usock->set_sock_buffsize(65535);
  usock->set_timer((float)connect_timer, 0.5f);
  if ((retval = usock->init()) != vvSocket::VV_OK)
  {
    delete usock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Error: RTT_server(): Timeout UDP init_client()");
    }
    else
    {
      vvDebugMsg::msg(1, "Error: RTT_server(): Socket could not be opened");
    }
    return -1;
  }
  frame = new uchar[payload];
  startTime(0);
  for ( ; ;)
  {
    if ((retval = usock->read_data(frame, payload)) == vvSocket::VV_OK)
    {
      if ((retval = usock->write_data(frame, 1)) != vvSocket::VV_OK)
      {
        delete usock;
        if  (retval == vvSocket::VV_TIMEOUT_ERROR)
        {
          vvDebugMsg::msg(1, "Error: RTT_server(): Timeout write");
        }
        else
        {
          vvDebugMsg::msg(1, "Error: RTT_server(): Writing data failed");
        }
        delete[] frame;
        return -1;
      }
    }
    else if(retval != vvSocket::VV_TIMEOUT_ERROR)
    {
      delete usock;
      vvDebugMsg::msg(1, "Error: RTT_server(): Reading data failed");
      delete[] frame;
      return -1;
    }
    if (getTime(0) > 2000)
      break;
  }
  delete usock;
  delete[] frame;
#else
  payload = payload;                              // prevent "unreferenced parameter"
#endif
  return 0;
}

//----------------------------------------------------------------------------
/**CLient for round-trip-time measurement. Needed for bandwidth-delay-product.
 NOT SUPPORTED UNDER WINDOWS AND WHEN THE VV_BDP FLAG IS NOT SET
 @param payload   payload size in bytes for UDP packets
*/
float vvSocket::RTT_client(int payload)
{
#if !defined(_WIN32) && defined(VV_BDP)
  uchar* frame;
  int valid_measures=0;
  float rtt;
  float sum=0;
  ErrorType retval;

  vvSocket* usock = new vvSocket(port, hostname, VV_UDP, cl_min_port, cl_max_port);
  usock->set_debuglevel(debuglevel);
  usock->set_sock_buffsize(65535);
  usock->set_timer((float)connect_timer, 0.5f);
  if ((retval = usock->init()) != vvSocket::VV_OK)
  {
    delete usock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Error: RTT_client(): Timeout UDP init_client()");
    }
    else
    {
      vvDebugMsg::msg(1, "Error: RTT_client(): Socket could not be opened");
    }
    return -1;
  }
  frame = new uchar[payload];
  startTime(0);
  for ( ; ; )
  {
    startTime(1);
    if ((retval = usock->write_data(frame, payload)) != vvSocket::VV_OK)
    {
      delete usock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Error: RTT_client(): Timeout write");
      }
      else
      {
        vvDebugMsg::msg(1, "Error: RTT_client(): Writing data failed");
      }
      delete[] frame;
      return -1;
    }
    if ((retval = usock->read_data(frame, 1)) == vvSocket::VV_OK)
    {
      valid_measures ++;
      rtt = getTime(1);
      sum += rtt;
    }
    else if (retval != vvSocket::VV_TIMEOUT_ERROR)
    {
      delete usock;
      vvDebugMsg::msg(1, "Error: RTT_client(): Reading data failed");
      delete[] frame;
      return -1;
    }
    if (getTime(0) > 2000)
      break;
  }
  delete usock;
  delete[] frame;
  if (valid_measures > 5)
  {
    rtt = sum/valid_measures;
    if (debuglevel>1)
    {
      ostringstream errmsg;
      errmsg << "average rtt: "<<rtt<<" ms";
      vvDebugMsg::msg(2, errmsg.str().c_str());
    }
    return rtt;
  }
  else
    return 0;
#else
  payload = payload;                              // prevent "unreferenced parameter"
  return 0;
#endif
}

//----------------------------------------------------------------------------
/**Server for Measurement of the bandwidth-delay-product. Socket
 buffers are set to the measured BDP if BDP is larger than default buffer sizes.
 NOT SUPPORTED UNDER WINDOWS AND WHEN THE VV_BDP FLAG IS NOT SET
*/
int vvSocket::measure_BDP_server()
{
#if !defined(_WIN32) && defined(VV_BDP)
  int pid, status;
  int pip[2];

  if (pipe(pip) < 0)
  {
    vvDebugMsg::msg(1, "Error pipe()");
    return -1;
  }
  if ((pid = fork()) < 0)
  {
    vvDebugMsg::msg(1, "Error fork()");
    return -1;
  }
  else if (pid == 0)
  {
    uchar* buffer;
    ErrorType retval;
    int  bdp, recvbdp, mtu, recvmtu;

    vvSocket* sock = new vvSocket(port, VV_TCP);
    sock->set_debuglevel(debuglevel);
    sock->set_sock_buffsize(1048575);
    sock->set_timer((float)connect_timer, transfer_timer);
    if ((retval = sock->init()) != vvSocket::VV_OK)
    {
      delete sock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Timeout connection establishment");
      }
      else
      {
        vvDebugMsg::msg(1, "Socket could not be opened");
      }
      exit(-1);
    }
    if ((retval = sock->read_data((uchar *)&recvmtu, 4)) != vvSocket::VV_OK)
    {

      delete sock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Timeout read");
      }
      else
      {
        vvDebugMsg::msg(1, "Reading data failed");
      }
      exit(-1);
    }
    mtu = ntohl(recvmtu);
    ostringstream errmsg;
    errmsg << "Received MTU: " << mtu << " bytes";
    vvDebugMsg::msg(1, errmsg.str().c_str());
    if (RTT_server(mtu - 28))
    {
      vvDebugMsg::msg(1, "error RTT_server()");
      exit(-1);
    }
    buffer = new uchar[1000000];
    for (int j=0; j< 3 ; j++)
    {
      if ((retval = sock->write_data(buffer, 1000000)) != vvSocket::VV_OK)
      {
        delete[] buffer;
        delete sock;
        if  (retval == vvSocket::VV_TIMEOUT_ERROR)
        {
          vvDebugMsg::msg(1, "Timeout write");
        }
        else
        {
          vvDebugMsg::msg(1, "Writing data failed");
        }
        exit(-1);
      }
    }
    delete[] buffer;
    if ((retval = sock->read_data((uchar *)&recvbdp, 4)) != vvSocket::VV_OK)
    {

      delete sock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Timeout read");
      }
      else
      {
        vvDebugMsg::msg(1, "Reading data failed");
      }
      exit(-1);
    }
    bdp = ntohl(recvbdp);
    ostringstream errmsg;
    errmsg << "Received bandwidth-delay-product: " << bdp << " bytes";
    vvDebugMsg::msg(1, errmsg.str().c_str();
    if (bdp < get_recv_buffsize())
      sock_buffsize = recv_buffsize;
    else
      sock_buffsize = bdp;
    delete sock;
    close(pip[0]);
    write(pip[1],(uchar *)&sock_buffsize, 4);
    exit(0);

  }
  else
  {
    if (waitpid(pid, &status, 0) != pid)
    {
      vvDebugMsg::msg(1, "error waitpid()");
      return -1;
    }
    if (status)
      return -1;
    close(pip[1]);
    read(pip[0],(uchar *)&sock_buffsize, 4);
  }
#endif
  return 0;
}

//----------------------------------------------------------------------------
/**CLient for Measurement of the bandwidth-delay-product. Socket
 buffers are set to the measured BDP if BDP is larger than default buffer sizes.
 NOT SUPPORTED UNDER WINDOWS AND WHEN THE VV_BDP FLAG IS NOT SET
*/
int vvSocket::measure_BDP_client()
{
#if !defined(_WIN32) && defined(VV_BDP)
  uchar* buffer;
  float time, rtt;
  int speed;
  ErrorType retval;
  int sum=0;
  int sendbdp, bdp, mtu, sendmtu;

  vvSocket* sock = new vvSocket(port, hostname, VV_TCP, cl_min_port, cl_max_port);
  sock->set_debuglevel(debuglevel);
  sock->set_sock_buffsize(1048575);
  sock->set_timer((float)connect_timer, 2.0f);
  if ((retval = sock->init()) != vvSocket::VV_OK)
  {
    delete sock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Timeout connect");
    }
    else
    {
      vvDebugMsg::msg(1, "Socket could not be opened");
    }
    return -1;
  }
  mtu = sock->get_MTU();
  sendmtu = htonl(mtu);
  if ((retval = sock->write_data((uchar *)&sendmtu, 4)) != vvSocket::VV_OK)
  {
    delete sock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Timeout write");
    }
    else
    {
      vvDebugMsg::msg(1, "Writing data failed");
    }
    return -1;
  }
  sleep(1);
  if ((rtt = RTT_client(mtu-28)) < 0)
  {
    vvDebugMsg::msg(1, "error get_RTT()");
    return -1;
  }
  buffer = new uchar[1000000];
  for (int j=0; j< 3 ; j++)
  {
    startTime(0);
    if ((retval = sock->read_data(buffer, 1000000)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      delete sock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Timeout read");
      }
      else
      {
        vvDebugMsg::msg(1, "Reading data failed");
      }
      return -1;
    }
    time =  getTime(0);
    if (j > 0)                                    // first measurement is a bad one
    {
      speed = (int)(8000000/time);
      sum += speed;
      ostringstream errmsg;
      errmsg << "speed: "<<speed<<" Kbit/s";
      vvDebugMsg::msg(3, errmsg.str().c_str();
    }
  }
  delete[] buffer;
  speed = sum/2;
  ostringstream errmsg;
  errmsg << "average speed: "<<speed<<" Kbit/s";
  vvDebugMsg::msg(2, errmsg.str().c_str());
  if (speed > 100000)
    speed = 1000000;
  else if (speed > 10000)
    speed = 100000;
  else if (speed > 1000)
    speed = 10000;
  bdp = (int)(speed * rtt)/8;
  errmsg.str("");
  errmsg << "bandwith-delay-product: "<<bdp<<" bytes";
  vvDebugMsg::msg(2, errmsg.str().c_str());
  sendbdp = htonl(bdp);
  if ((retval = sock->write_data((uchar *)&sendbdp, 4)) != vvSocket::VV_OK)
  {
    delete sock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Timeout write");
    }
    else
    {
      vvDebugMsg::msg(1, "Writing data failed");
    }
    return -1;
  }
  if (bdp < get_recv_buffsize())
    sock_buffsize = recv_buffsize;
  else
    sock_buffsize = bdp;

  delete sock;
  sleep(1);                                       //give server time to listen
#endif
  return 0;
}

//----------------------------------------------------------------------------
/**Tries to determine the MTU. Connection must be established for getting the
real value.
*/
int vvSocket::get_MTU()
{

#ifndef TCP_MAXSEG
  vvDebugMsg::msg(2, "TCP_MAXSEG is not defined, use 576 bytes for MTU");
  return 576;
#else
  int rc;
  int theMSS = 0;
  socklen_t len = sizeof( theMSS );
  rc = getsockopt( sockfd, IPPROTO_TCP, TCP_MAXSEG, (char*) &theMSS, &len );
  if(rc == -1 || theMSS <= 0)
  {
    vvDebugMsg::msg(2, "OS doesn't support TCP_MAXSEG querry? use 576 bytes for MTU");
    return 576;
  }
  else if ( checkMSS_MTU( theMSS, 1500 ))
  {
    vvDebugMsg::msg(2, "ethernet, mtu=1500 bytes");
    return 1500;
  }
  else if ( checkMSS_MTU( theMSS, 4352 ))
  {
    vvDebugMsg::msg(2, "FDDI, mtu=4352 bytes");
    return 4352;
  }
  else if ( checkMSS_MTU( theMSS, 9180 ))
  {
    vvDebugMsg::msg(2, "ATM, mtu=9180 bytes");
    return 9180;
  }
  else if ( checkMSS_MTU( theMSS, 65280 ))
  {
    vvDebugMsg::msg(2, "HIPPI, mtu=65280 bytes");
    return 65280;
  }
  else
  {
    ostringstream errmsg;
    errmsg << "unknown interface, mtu set to " << theMSS+40 << " bytes";
    vvDebugMsg::msg(2, errmsg.str().c_str());
    return  theMSS + 40;
  }
#endif
}

//----------------------------------------------------------------------------
/**Checks if the MSS belongs to a well-known MTU
 @param mss   given MSS
 @param mtu   MTU to check
*/
int vvSocket::checkMSS_MTU(int mss, int mtu)
{
  return (mtu-40) >= mss  &&  mss >= (mtu-80);
}

//----------------------------------------------------------------------------
/** With this function the linger option can be turned on. As default the linger option
  is turned off.
  @param sec   number of seconds for linger time.
*/
int vvSocket::set_linger(int sec)
{
  struct linger ling;

  ling.l_onoff = 1;
  ling.l_linger = (unsigned short)sec;
  if (setsockopt(sockfd, SOL_SOCKET, SO_LINGER, (char*)&ling, sizeof(ling)))
  {
    vvDebugMsg::msg(1, "Error: setsockopt()");
    return -1;
  }
  return 0;
}

//----------------------------------------------------------------------------
/** Disables TCP's Nagle algorithm
 */
int vvSocket::no_nagle()
{
  int on=1;
  if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY , (char*)&on, sizeof(on)))
  {
    vvDebugMsg::msg(1, "Error: setsockopt()");
    return -1;
  }
  return 0;

}

//----------------------------------------------------------------------------
/// Initialize a UDP server
vvSocket::ErrorType vvSocket::init_server_udp()
{
#ifdef _WIN32
  char optval=1;
#else
  int optval=1;
#endif
  ErrorType retval;
  float temp;
  if ((sockfd= socket(AF_INET, SOCK_DGRAM, 0)) <0)
  {
    vvDebugMsg::msg(1, "Error: socket()");
    return VV_SOCK_ERROR;
  }
  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval,sizeof(optval)))
  {
    vvDebugMsg::msg(1, "Error: setsockopt()");
    return VV_SOCK_ERROR;
  }
  if (sock_buffsize > 0)
  {
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF,
      (char *) &sock_buffsize, sizeof(sock_buffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF,
      (char *) &sock_buffsize, sizeof(sock_buffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
  }
  get_send_buffsize();
  ostringstream errmsg;
  errmsg << "send_buffsize: " << send_buffsize << " bytes, recv_buffsize: " << get_recv_buffsize() << " bytes";
  vvDebugMsg::msg(2, errmsg.str().c_str());
  if (send_buffsize < 65507)
    max_send_size = send_buffsize;
  else
    max_send_size = 65507;
  memset((char *) &host_addr, 0, sizeof(host_addr));
  host_addr.sin_family = AF_INET;
  host_addr.sin_port = htons((unsigned short)port);
  host_addr.sin_addr.s_addr = INADDR_ANY;
  host_addrlen = sizeof(host_addr);

  if  (bind(sockfd, (struct sockaddr *)&host_addr, host_addrlen))
  {
    vvDebugMsg::msg(1, "Error: bind()");
    return VV_SOCK_ERROR ;
  }
  temp = transfer_timer;
  transfer_timer= (float)connect_timer;
  if ((retval = get_client_addr()) != VV_OK)
  {
    if  (retval == VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Timeout get_client_addr()");
      return VV_TIMEOUT_ERROR;
    }
    else
    {
      vvDebugMsg::msg(1, "Error: get_client_addr()");
      return VV_READ_ERROR;
    }
  }
  transfer_timer = temp;
  if (connect(sockfd, (struct sockaddr *)&host_addr, host_addrlen))
  {
    vvDebugMsg::msg(1, "Error: connect()");
    return VV_CONNECT_ERROR;
  }
  return VV_OK;
}

//----------------------------------------------------------------------------
/// Initialize a UDP client
vvSocket::ErrorType vvSocket::init_client_udp()
{
  int cl_port;
  ErrorType retval;
  uchar buff;

#ifdef _WIN32
  host= gethostbyname(hostname);
  if (host == 0)
  {
    vvDebugMsg::msg(1, "Error gethostbyname()");
    return VV_HOST_ERROR;
  }
#else
  Sigfunc *sigfunc;

  sigfunc = Signal(SIGALRM, nonameserver);
  if (alarm(5) != 0)
  {
    vvDebugMsg::msg(2, "init_client():WARNING! previously set alarm was wiped out");
  }
  if ((host= gethostbyname(hostname)) == 0)
  {
    alarm(0);
    vvDebugMsg::msg(1, "Error: gethostbyname()");
    signal(SIGALRM, sigfunc);
    return VV_HOST_ERROR;
  }
  alarm(0);
  signal(SIGALRM, sigfunc);
#endif

  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0)
  {
    vvDebugMsg::msg(1, "Error: socket");
    return VV_SOCK_ERROR;
  }
  if (sock_buffsize > 0)
  {
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF,
      (char *) &sock_buffsize, sizeof(sock_buffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF,
      (char *) &sock_buffsize, sizeof(sock_buffsize)))
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
  }
  get_send_buffsize();
  ostringstream errmsg;
  errmsg << "send_buffsize: " << send_buffsize << "bytes,  recv_buffsize: " << get_recv_buffsize() << "bytes";
  vvDebugMsg::msg(1, errmsg.str().c_str());
  if (send_buffsize < 65507)
    max_send_size = send_buffsize;
  else
    max_send_size = 65507;
  memset((char *) &host_addr, 0, sizeof(host_addr));
  host_addr.sin_family = AF_INET;
  host_addrlen = sizeof(host_addr);
  if (cl_min_port != 0 || cl_max_port != 0)
  {
    if (cl_min_port > cl_max_port)
    {
      vvDebugMsg::msg(1, "Wrong port range");
      return VV_SOCK_ERROR ;
    }
    host_addr.sin_addr.s_addr = INADDR_ANY;
    cl_port = cl_min_port;
    host_addr.sin_port = htons((unsigned short)cl_port);
    while (bind(sockfd, (struct sockaddr *)&host_addr, host_addrlen) && cl_port <= cl_max_port)
    {
#ifdef _WIN32
      if (WSAGetLastError() == WSAEADDRINUSE)
#else
        if (errno == EADDRINUSE)
#endif
      {
        cl_port ++;
        host_addr.sin_port = htons((unsigned short)cl_port);
      }
      else
      {
        vvDebugMsg::msg(1, "Error: bind()");
        return VV_SOCK_ERROR ;
      }
    }
    if (cl_port > cl_max_port)
    {
      vvDebugMsg::msg(1, "No port free!");
      return VV_SOCK_ERROR ;
    }
  }
  host_addr.sin_addr = *((struct in_addr *)host->h_addr);
  host_addr.sin_port = htons((unsigned short)port);
  if (connect(sockfd, (struct sockaddr *)&host_addr, host_addrlen))
  {
    vvDebugMsg::msg(1, "Error: connect()");
    return VV_CONNECT_ERROR;
  }
  if ((retval = write_data(&buff, 1)) != VV_OK)
  {
    if  (retval == VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Timeout: write_data()");
      return VV_TIMEOUT_ERROR;
    }
    else
    {
      vvDebugMsg::msg(1, "Error: write_data()");
      return VV_WRITE_ERROR;
    }
  }
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Reads a message from the client to get his address for a connected UDP socket.
 Calls recvfrom_timeo() for reading with timeout and recvfrom_nontimeo()
 for reading without a timeout.
*/
vvSocket::ErrorType vvSocket::get_client_addr()
{

  ErrorType retval;

  if (connect_timer > 0)
  {
    do
    retval = recvfrom_timeo();
    while (retval == VV_RETRY);
  }
  else
    retval = recvfrom_nontimeo();
  return retval;

}

//----------------------------------------------------------------------------
/** Reads a message from the client with a timeout.
 */
vvSocket::ErrorType vvSocket::recvfrom_timeo()
{
  uchar buff;

  vvDebugMsg::msg(3, "waiting .....");
  if (readable_timeo())
  {
    if(recvfrom(sockfd, (char*)&buff, 1, 0,(struct sockaddr *)&host_addr, &host_addrlen) !=1)
    {
      vvDebugMsg::msg(1, "Error: recvfrom()");
      return VV_READ_ERROR;
    }
    vvDebugMsg::msg(3, "Client Address received");
  }
  else
  {
    if (vvDebugMsg::getDebugLevel()>1)
      if (do_a_retry())
        return VV_RETRY;
    return VV_TIMEOUT_ERROR;
  }
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Reads a message from the client without a timeout.
 */
vvSocket::ErrorType vvSocket::recvfrom_nontimeo()
{

  uchar buff;

  if(recvfrom(sockfd, (char*)&buff, 1, 0,(struct sockaddr *)&host_addr, &host_addrlen) !=1)
  {
    vvDebugMsg::msg(1, "Error: recvfrom()");
    return VV_READ_ERROR;
  }
  vvDebugMsg::msg(3, "Client Address received");
  return VV_OK;
}

//----------------------------------------------------------------------------
/**Reads data from the UDP socket.
 @param buffer  pointer to the data to write
 @param size   number of bytes to write

*/
ssize_t vvSocket::readn_udp(char* buffer, size_t size)
{
  size_t nleft;
  ssize_t nread;

  nleft = size;
  while(nleft > 0)
  {
    nread = recv(sockfd, buffer, nleft, 0);
    if (nread < 0)
    {
#ifdef _WIN32
      if (WSAGetLastError() == WSAEINTR)
#else
        if (errno == EINTR)
#endif
          nread = 0;                              // interrupted, call read again
      else
      {
        vvDebugMsg::msg(1, "Error: udp recv()");
        return (ssize_t)-1;
      }
    }
    else if (nread == 0)
      break;
    ostringstream errmsg;
    errmsg <<  nread << " Bytes read";
    vvDebugMsg::msg(3, errmsg.str().c_str());
    nleft -= nread;
    buffer += nread;
  }
  return (size - nleft);
}

//----------------------------------------------------------------------------
/**Writes data to the UDP socket.
 @param buffer  pointer to the data to write
 @param size   number of bytes to write

*/
ssize_t vvSocket::writen_udp(const char* buffer, size_t size)
{
  size_t nleft, towrite;
  ssize_t nwritten;

  nleft = size;

  while(nleft > 0)
  {
    if (nleft > (size_t)max_send_size)
      towrite = max_send_size;
    else
      towrite = nleft;
    nwritten = send(sockfd, buffer, towrite, 0);
    if (nwritten < 0)
    {
#ifdef _WIN32
      if (WSAGetLastError() == WSAEINTR)
#else
        if (errno == EINTR)
#endif
          nwritten = 0;                           // interrupted, call write again
      else
      {
        vvDebugMsg::msg(1, "Error: udp send()");
        return (ssize_t)-1;
      }
    }
    ostringstream errmsg;
    errmsg << nwritten << " Bytes written";
    vvDebugMsg::msg(3, errmsg.str().c_str());
    nleft -= nwritten;
    buffer += nwritten;
  }
  return size;
}

//----------------------------------------------------------------------------
/** Constructor for server
 @param portnumber   port number
 @param st           socket type
*/
vvSocket::vvSocket(const unsigned short portnumber, SocketType st)
:  port(portnumber),socktype(st)
{
  hostname = 0;
  sockfd = -1;
  initVars();
}

//----------------------------------------------------------------------------
/** Constructor for client
 @param portnumber   port number
 @param servername   name of server to connect to
 @param st           socket type
 @param clminport    minimum outgoing port
 @param clmaxport    maximum outgoing port
*/
vvSocket::vvSocket(const unsigned short portnumber, const char* servername,
                   SocketType st, int clminport, int clmaxport)
:  port(portnumber),   hostname(servername),socktype(st),cl_min_port(clminport),cl_max_port(clmaxport)
{
  sockfd = -1;
  initVars();
}

//----------------------------------------------------------------------------
/** Constructor from already accept() 'ed socket file descriptor.
 @param sockfd       socket file descriptor
 @param st           socket type
*/
vvSocket::vvSocket(const SocketType socktype, const int sockfd)
  :socktype(socktype)
{
  port = -1;
  hostname = NULL;
  this->sockfd = sockfd;
  initVars();
}

//----------------------------------------------------------------------------
/// Destructor
vvSocket::~vvSocket()
{
#ifdef _WIN32
  if(sockfd >= 0)
    if(closesocket(sockfd))
      if (WSAGetLastError() ==  WSAEWOULDBLOCK)
        vvDebugMsg::msg(1, "Linger time expires");
  WSACleanup();
#else
  if(sockfd >= 0)
    if (close(sockfd))
      if (errno ==  EWOULDBLOCK)
        vvDebugMsg::msg(1, "Linger time expires");
#endif
}

//----------------------------------------------------------------------------
/** signal function for timeouts
 @param signo
 @param func
*/
vvSocket::Sigfunc *vvSocket::signal(int signo, vvSocket::Sigfunc *func)
{
#ifdef _WIN32
  signo = 0;
  func = NULL;
#else
  struct sigaction  act, oact;

  act.sa_handler = func;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;
  if (signo == SIGALRM)
  {
#ifdef  SA_INTERRUPT
    act.sa_flags |= SA_INTERRUPT;                 // SunOS 4.x
#endif
  }
  else
  {
#ifdef  SA_RESTART
    act.sa_flags |= SA_RESTART;                   // SVR4, 44BSD
#endif
  }
  if (sigaction(signo, &act, &oact) >= 0)
    return(oact.sa_handler);
#endif
  return(SIG_ERR);
}

//----------------------------------------------------------------------------
/** Signal function for timeouts
 @param signo
 @param func
*/
                                                  /* for signal() function */
vvSocket::Sigfunc *vvSocket::Signal(int signo, vvSocket::Sigfunc *func)
{
  Sigfunc *sigfunc;

  if ( (sigfunc = signal(signo, func)) == SIG_ERR)
    vvDebugMsg::msg(1, "signal error");
  return(sigfunc);
}

//----------------------------------------------------------------------------
/** Error message if there's no nameserver available.
 */
void vvSocket::nonameserver(int)
{
  cerr<<"Nameserver not found. Contact your system administrator! Waiting for timeout..."<<endl;
  return;
}

//----------------------------------------------------------------------------
/** Error message if SIGPIPE is issued on write() / peer closed the connection.
 */
void vvSocket::peerunreachable(int)
{
  cerr<<"Caught signal SIGPIPE. Peer unreachable."<<endl;
  return;
}

//----------------------------------------------------------------------------
/** Checks if there's data on the socket to read, for a time of length
 transfer_timer.
 @return -1 for error, 0 for timeout, >0 for data to read
*/
int vvSocket::readable_timeo()
{
  fd_set rset;
  struct timeval tv;

  FD_ZERO(&rset);
  FD_SET((unsigned int)sockfd, &rset);

  tv.tv_sec = (int)transfer_timer;
  tv.tv_usec = (int)(1000000*(transfer_timer - (int)transfer_timer));

  return (select(sockfd +1, &rset, 0, 0, &tv));
}

//----------------------------------------------------------------------------
/** Checks if data can be written to the socket, for a time of length
 transfer_timer.
 @return -1 for error, 0 for timeout, >0 if data can be written.
*/
int vvSocket::writeable_timeo()
{
  fd_set rset;
  struct timeval tv;

  FD_ZERO(&rset);
  FD_SET((unsigned int)sockfd, &rset);

  tv.tv_sec  = (int)transfer_timer;
  tv.tv_usec = (int)(1000000*(transfer_timer - (int)transfer_timer));

  return (select(sockfd +1, 0, &rset, 0, &tv));
}

//----------------------------------------------------------------------------
/** Reads from a socket with a timeout. Calls readn_tcp() for TCP sockets
and readn_udp() for UDP Sockets.
 @param dataptr  pointer to where the read data is written.
 @param size  number of bytes to read
*/
vvSocket::ErrorType vvSocket::read_timeo(uchar* dataptr, size_t size)
{
  ssize_t s;

  if (size <= 0) return VV_OK;

  vvDebugMsg::msg(3, "waiting .....");
  if (readable_timeo())
  {
    if (socktype == VV_TCP)
      s = readn_tcp((char*)dataptr, size);
    else
      s = readn_udp((char*)dataptr, size);
    if (s == -1)
    {
      vvDebugMsg::msg(1, "Reading data failed, read_timeo()");
      return VV_READ_ERROR;
    }
    ostringstream errmsg;
    errmsg << "Getting " << s << " Bytes of Data";
    vvDebugMsg::msg(3, errmsg.str().c_str());
    return VV_OK;
  }
  else
  {
    if (vvDebugMsg::getDebugLevel() > 1)
      if (do_a_retry())
        return VV_RETRY;
    return VV_TIMEOUT_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Reads from a socket without a timeout. Calls readn_tcp() for TCP sockets
and readn_udp() for UDP Sockets.
 @param dataptr  pointer to where the read data is written.
 @param size  number of bytes to read
*/
vvSocket::ErrorType vvSocket::read_nontimeo(uchar* dataptr, size_t size)
{
  ssize_t s;

  if (size <= 0) return VV_OK;

  vvDebugMsg::msg(3, "waiting .....");
  if (socktype == VV_TCP)
    s = readn_tcp((char*)dataptr, size);
  else
    s = readn_udp((char*)dataptr, size);
  if (s == -1)
  {
    vvDebugMsg::msg(1, "Reading data failed, read_nontimeo()");
    return VV_READ_ERROR;
  }
  if (s == 0)
  {
    vvDebugMsg::msg(1, "Peer performed orderly shutdown");
    return VV_PEER_SHUTDOWN;
  }
  ostringstream errmsg;
  errmsg << "Getting " << s << " Bytes of Data";
  vvDebugMsg::msg(3, errmsg.str().c_str());
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Writes to a socket with a timeout.Calls writen_tcp() for TCP Sockets
 and writen_udp() for UDP sockets.
 @param dataptr  pointer to the data to write.
 @param size  number of bytes to write.
*/
vvSocket::ErrorType vvSocket::write_timeo(const uchar* dataptr, size_t size)
{
  ssize_t  s;

  if (writeable_timeo())
  {
    if (socktype == VV_TCP)
      s = writen_tcp((char*)dataptr, size);
    else
      s = writen_udp((char*)dataptr, size);
    if (s == -1)
    {
      vvDebugMsg::msg(1, "Writing data failed, write_timeo()");
      return VV_WRITE_ERROR;
    }
    ostringstream errmsg;
    errmsg << "Sending " << s << " Bytes of Data";
    vvDebugMsg::msg(3, errmsg.str().c_str());
    return VV_OK;
  }
  else
  {
    if (vvDebugMsg::getDebugLevel() > 1)
      if (do_a_retry())
        return VV_RETRY;
    return VV_TIMEOUT_ERROR;
  }
}

//----------------------------------------------------------------------------
/** Writes to a socket without a timeout. Calls writen_tcp() for TCP Sockets
 and writen_udp() for UDP sockets.
 @param dataptr  pointer to the data to write.
 @param size  number of bytes to write.
*/
vvSocket::ErrorType vvSocket::write_nontimeo(const uchar* dataptr, size_t size)
{
  ssize_t  s;

  if (socktype == VV_TCP)
    s = writen_tcp((char*)dataptr, size);
  else
    s = writen_udp((char*)dataptr, size);
  if (s == -1)
  {
    vvDebugMsg::msg(1, "Writing data failed, write_nontimeo()");
    return VV_WRITE_ERROR;
  }
  ostringstream errmsg;
  errmsg << "Sending " << s << " Bytes of Data";
  vvDebugMsg::msg(3, errmsg.str().c_str());
  return VV_OK;

}

//----------------------------------------------------------------------------
/** Turns blocking status of sockets on and off. (default: blocking on)
*/
vvSocket::ErrorType vvSocket::set_nonblocking(bool on)
{
#ifndef _WIN32
  int flags = fcntl(sockfd, F_GETFL, 0);
  if(flags < 0)
  {
    vvDebugMsg::msg(2, "Getting flags of socket failed");
    return VV_SOCK_ERROR;
  }
#endif
  if (on)
  {
#ifdef _WIN32
    unsigned long tru = 1;
    ioctlsocket(sockfd, FIONBIO, &tru);
#else
    if(fcntl(sockfd, F_SETFL, flags|O_NONBLOCK))
    {
      vvDebugMsg::msg(2, "setting O_NONBLOCK on socket failed");
      return VV_OK;
    }
#endif
  }
  else
  {
#ifdef _WIN32
    unsigned long tru = 0;
    ioctlsocket(sockfd, FIONBIO, &tru);
#else
    if(fcntl(sockfd, F_SETFL, flags & (~O_NONBLOCK)))
    {
      vvDebugMsg::msg(2, "removing O_NONBLOCK from socket failed.");
      return VV_SOCK_ERROR;
    }
#endif
  }
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Initializes a socket. Has to be called after the creation
 of an socket class instance. Calls init_client() for client and init_server()
 for server if socket wasn't constructed from an already initialized sockfd.
*/
vvSocket::ErrorType vvSocket::init()
{
  if (sockfd != -1)
  {
    vvDebugMsg::msg(1, "Error: called init() on initialized socket()");
    return VV_SOCKFD_ERROR;
  }

  if (hostname)
  {
    switch(socktype)
    {
    case VV_TCP:
      return init_client_tcp();
    case VV_UDP:
      return init_client_udp();
    case VV_MC_SENDER:
      return init_client_mc();
    case VV_MC_RECEIVER:
      return init_server_mc();
    }
  }
  else
  {
    switch(socktype)
    {
    case VV_TCP:
      return init_server_tcp();
    case VV_UDP:
      return init_server_udp();
    case VV_MC_SENDER:
    case VV_MC_RECEIVER:
      vvDebugMsg::msg(2, "Multicasting is not available without a host address");
      return VV_SOCK_ERROR;
    }
  }
  // compiler thinks end of function is reachable
  return VV_SOCK_ERROR;
}

//----------------------------------------------------------------------------
/** Function to read data from the socket.
 Calls read_timeo() for reading with timeout and read_nontimeo() for
 reading without timeout.
 @param dataptr  pointer to where the read data is written.
 @param size  number of bytes to read
*/
vvSocket::ErrorType vvSocket::read_data(uchar* dataptr, size_t size)
{
  if(VV_MC_RECEIVER == socktype)
  {
    ssize_t got = read(sockfd, dataptr, size);
    if(got == size)
    {
      return VV_OK;
    }
    else
    {
      vvDebugMsg::msg(2, "vvSocket::read_data()", true);
      return VV_WRITE_ERROR;
    }
  }
  else if(VV_MC_SENDER == socktype)
  {
    return VV_SOCKTYPE_ERROR;
  }

  ErrorType retval;

  if (transfer_timer > 0)
  {
    do
    retval = read_timeo(dataptr, size);
    while (retval == VV_RETRY);
  }
  else
    retval = read_nontimeo(dataptr, size);
  return retval;
}

//----------------------------------------------------------------------------
/** Function to write data to the socket.
 Calls write_timeo() for writing with timeout and write_nontimeo() for
 writing without timeout.
 @param dataptr  pointer to the data to write.
 @param size  number of bytes to write.
*/
vvSocket::ErrorType vvSocket::write_data(const uchar* dataptr, size_t size)
{
  if(VV_MC_SENDER == socktype)
  {
    ssize_t written = sendto(sockfd, dataptr, size, 0, (struct sockaddr*)&groupSock, sizeof(groupSock));
    if(written == size)
    {
      return VV_OK;
    }
    else
    {
      vvDebugMsg::msg(2, "vvSocket::write_data()", true);
      return VV_WRITE_ERROR;
    }
  }
  else if(VV_MC_RECEIVER == socktype)
  {
    return VV_SOCKTYPE_ERROR;
  }

  ErrorType retval;

  if (transfer_timer > 0)
  {
    do
    retval = write_timeo(dataptr, size);
    while (retval == VV_RETRY);
  }
  else
    retval = write_nontimeo(dataptr, size);
  return retval;
}

//----------------------------------------------------------------------------
/** Returns the number of bytes currently in the socket receive buffer.
 */
int vvSocket::is_data_waiting() const
{
#ifdef _WIN32
  unsigned long nbytes;
#else
  size_t nbytes;
#endif

#ifdef _WIN32
  if(ioctlsocket(sockfd, FIONREAD, &nbytes))
  {
    vvDebugMsg::msg(1, "Error: ioctlsocket()");
    return -1;
  }
#else
  if(ioctl(sockfd, FIONREAD, &nbytes))
  {
    vvDebugMsg::msg(1, "Error: ioctl()");
    return -1;
  }
#endif
  return nbytes;
}

//----------------------------------------------------------------------------
/** Sets the socket buffer size for sending and receiving.
 Automatic buffer size measurement for sbs=0. Has to be called before
 the init() function.
 @param sbs  desired socket buffsize in bytes.
*/
void vvSocket::set_sock_buffsize(int sbs)
{
  sock_buffsize=sbs;
}

//----------------------------------------------------------------------------
/** Sets the timers for the connection establishment and for the data transfer.
 No timers for a value equals to zero (default).
 @param ct  timer for connection estabilishment in sec.
 @param tt  timer for data transfer in sec.
*/
void vvSocket::set_timer(float ct, float tt)
{
  if (ct>0.0f && ct<1.0f)
    ct = 1.0f;
  connect_timer = int(ct);
  transfer_timer = tt;
}

//----------------------------------------------------------------------------
/** Returns the socket file descriptor.
 */
int vvSocket::get_sockfd() const
{
  return sockfd;
}

void vvSocket::set_sockfd(int fd)
{
  sockfd = fd;
}

//----------------------------------------------------------------------------
/** Returns the actual socket receive buffer size.
 */
int vvSocket::get_recv_buffsize()
{
  if (getsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char *) &recv_buffsize, &bufflen))
  {
    vvDebugMsg::msg(1, "Error: getsockopt()");
    return -1;
  }
  return recv_buffsize;
}

//----------------------------------------------------------------------------
/** Returns the actual socket send buffer size.
 */
int vvSocket::get_send_buffsize()
{
  if (getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, (char *) &send_buffsize, &bufflen))
  {
    vvDebugMsg::msg(1, "Error: getsockopt()");
    return -1;
  }
  return send_buffsize;
}

//---------------------------------------------------------------------------
/** Reads a string (i.e. one line) from the socket.
 @param s  pointer to where the string is written.
 @param maxLen maximum length of string to read.
 Reads at most maxLen-1 characters from the socket,
 the last character is used for '\0' termination.
 Returns OK if maxLen characters were sufficient, otherwise RETRY.
*/
vvSocket::ErrorType vvSocket::read_string(char* s, int maxLen)
{
  int len = 0;
  bool done = false;

  while (len<maxLen-1 && !done)
  {
    read_data((uchar*)(&s[len]), 1);
    if (s[len]=='\n') done = true;
    ++len;
  }
  if(len < maxLen)
    s[len] = '\0';

  if(done)
    return VV_OK;
  else
    return VV_RETRY;
}

//---------------------------------------------------------------------------
/** Writes a string to the socket.
 @param s pointer to the string to write.
*/
vvSocket::ErrorType vvSocket::write_string(const char* s)
{
  ErrorType ret;
  int len = strlen(s);
  char* stemp = new char[len + 1];

  strcpy(stemp, s);
  stemp[len] = '\n';
  ret = write_data((uchar*)stemp, len+1);
  delete[] stemp;
  return ret;
}

//---------------------------------------------------------------------------
/** Decides by an input from the user if a timed out function
   shall be retried.
*/
bool vvSocket::do_a_retry()
{
  char in[32];

  vvDebugMsg::msg(1, "Timeout. Retry? (y/n)");
  cin >> in;
  if (in[0] == 'y' || in[0] ==  'Y')
    return true;
  else
    return false;
}

//---------------------------------------------------------------------------
/** Starts a timer.
 @param timer  index of the timer.
*/
void vvSocket::startTime(int timer)
{
  assert(timer<NUM_TIMERS);
#ifdef _WIN32
  t_start[timer] = clock();
#else
  gettimeofday(&t_start[timer], NULL);
#endif
}

//---------------------------------------------------------------------------
/** Returns current time relative to start time in milliseconds.
 @param timer  index of the timer.
*/
float vvSocket::getTime(int timer)
{
#ifdef _WIN32
  clock_t t_end;
  t_end = clock();
  return (float(t_end) - float(t_start[timer])) / float(CLOCKS_PER_SEC) * 1000.0f;
#else
  timeval t_end;
  gettimeofday(&t_end, NULL);
  return ((t_end.tv_sec - t_start[timer].tv_sec) * 1000.0f +
    (t_end.tv_usec - t_start[timer].tv_usec) / 1000.0f);
#endif
}

//---------------------------------------------------------------------------
/** Prints an error message.
 @param prefix  prefix for identifying the error place.
*/
void vvSocket::printErrorMessage(const char* prefix) const
{
  if (prefix==NULL)
    cerr << "Socket error: ";
  else cerr << prefix << ": ";

#ifdef _WIN32
  int errno;
  errno = WSAGetLastError();
#endif
  cerr << strerror(errno) << " (" << errno << ")" << endl;
}

//---------------------------------------------------------------------------
/** Reads a one byte value from the socket
 */
uchar vvSocket::read8()
{
  uchar value;
  read_data(&value, 1);
  return value;
}

//---------------------------------------------------------------------------
/** Writes a one byte value to the socket
 @param value  the byte to write
*/
vvSocket::ErrorType vvSocket::write8(uchar value)
{
  return write_data(&value, 1);
}

//---------------------------------------------------------------------------
/** Reads a two byte value with given endianess from the socket
 @param end  endianess
*/
ushort vvSocket::read16(vvSocket::EndianType end)
{
  uchar buf[2];
  vvSocket::read_data(buf, 2);
  if (end == VV_LITTLE_END)
    return ushort((int)buf[0] + (int)buf[1] * (int)256);
  else
    return ushort((int)buf[0] * (int)256 + (int)buf[1]);
}

//---------------------------------------------------------------------------
/** Writes a two byte value in the given endianess to the socket
 @param value  two byte value to write
 @param end  endianess
*/
vvSocket::ErrorType vvSocket::write16(ushort value, vvSocket::EndianType end)
{
  uchar buf[2];
  if (end == VV_LITTLE_END)
  {
    buf[0] = (uchar)(value & 0xFF);
    buf[1] = (uchar)(value >> 8);
  }
  else
  {
    buf[0] = (uchar)(value >> 8);
    buf[1] = (uchar)(value & 0xFF);
  }
  return vvSocket::write_data(buf, 2);
}

//---------------------------------------------------------------------------
/** Reads a four byte value with given endianess from the socket
 @param end  endianess
*/
uint vvSocket::read32(vvSocket::EndianType end)
{
  uchar buf[4];
  vvSocket::read_data(buf, 4);
  if (end == VV_LITTLE_END)
  {
    return uint((ulong)buf[3] * (ulong)16777216 + (ulong)buf[2] * (ulong)65536 +
      (ulong)buf[1] * (ulong)256 + (ulong)buf[0]);
  }
  else
  {
    return uint((ulong)buf[0] * (ulong)16777216 + (ulong)buf[1] * (ulong)65536 +
      (ulong)buf[2] * (ulong)256 + (ulong)buf[3]);
  }
}

//---------------------------------------------------------------------------
/** Writes a four byte value in the given endianess to the socket
 @param value  four byte value to write
 @param end  endianess
*/
vvSocket::ErrorType vvSocket::write32(uint value, vvSocket::EndianType end)
{
  uchar buf[4];
  if (end == VV_LITTLE_END)
  {
    buf[0] = (uchar)(value & 0xFF);
    buf[1] = (uchar)((value >> 8)  & 0xFF);
    buf[2] = (uchar)((value >> 16) & 0xFF);
    buf[3] = (uchar)(value  >> 24);
  }
  else
  {
    buf[0] = (uchar)(value  >> 24);
    buf[1] = (uchar)((value >> 16) & 0xFF);
    buf[2] = (uchar)((value >> 8)  & 0xFF);
    buf[3] = (uchar)(value & 0xFF);
  }
  return vvSocket::write_data(buf, 4);
}

//---------------------------------------------------------------------------
/** Reads a float with given endianess from the socket
 @param end  endianess
*/
float vvSocket::readFloat(vvSocket::EndianType end)
{
  uchar buf[4];
  float  fval;
  uchar* ptr;
  uchar  tmp;

  assert(sizeof(float) == 4);
  vvSocket::read_data(buf, 4);
  memcpy(&fval, buf, 4);
  if (getEndianness() != end)
  {
    // Reverse byte order:
    ptr = (uchar*)&fval;
    tmp = ptr[0]; ptr[0] = ptr[3]; ptr[3] = tmp;
    tmp = ptr[1]; ptr[1] = ptr[2]; ptr[2] = tmp;
  }
  return fval;
}

//---------------------------------------------------------------------------
/** Writes a float in the given endianess to the socket
 @param value  float to write
 @param end  endianess
*/
vvSocket::ErrorType vvSocket::writeFloat(float value, vvSocket::EndianType end)
{
  uchar buf[4];
  uchar tmp;

  memcpy(buf, &value, 4);
  if (getEndianness() != end)
  {
    // Reverse byte order:
    tmp = buf[0]; buf[0] = buf[3]; buf[3] = tmp;
    tmp = buf[1]; buf[1] = buf[2]; buf[2] = tmp;
  }
  return vvSocket::write_data(buf, 4);
}

//----------------------------------------------------------------------------
/** Returns the current system's endianness.
 */
vvSocket::EndianType vvSocket::getEndianness()
{
  float one = 1.0f;                               // memory representation of 1.0 on big endian machines: 3F 80 00 00
  uchar* ptr;

  ptr = (uchar*)&one;
  if (*ptr == 0x3f)
    return VV_BIG_END;
  else
  {
    assert(*ptr == 0);
    return VV_LITTLE_END;
  }
}

//----------------------------------------------------------------------------
/** Called from constructor.
 */
void vvSocket::initVars()
{
  sock_buffsize = -1;
  transfer_timer = 0.0;                           // No timeouts as default
  connect_timer = 0;
  bufflen = sizeof(send_buffsize);
#ifdef _WIN32
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2,0), &wsaData) != 0)
    vvDebugMsg::msg(1, "WSAStartup failed!");
#endif
}

//----------------------------------------------------------------------------
/// Initializes a multicast client (sender)
vvSocket::ErrorType vvSocket::init_client_mc()
{
  vvDebugMsg::msg(3, "Enter vvSocket::init_client_mc()");

  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0)
  {
    vvDebugMsg::msg(1, "Error: socket", true);
    return VV_SOCK_ERROR;
  }
  if (sock_buffsize > 0)
  {
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, (char *) &sock_buffsize, sizeof(sock_buffsize)) < 0)
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char *) &sock_buffsize, sizeof(sock_buffsize)) < 0)
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
  }
  get_send_buffsize();
  ostringstream errmsg;
  errmsg << "send_buffsize: " << send_buffsize << "bytes,  recv_buffsize: " << get_recv_buffsize() << "bytes";
  vvDebugMsg::msg(1, errmsg.str().c_str());
  if (send_buffsize < 65507)
    max_send_size = send_buffsize;
  else
    max_send_size = 65507;

  memset((char *) &groupSock, 0, sizeof(groupSock));
  groupSock.sin_family = AF_INET;
  groupSock.sin_addr.s_addr = inet_addr(hostname);
  groupSock.sin_port = htons(port);

  char loopch = 0;
  if(setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_LOOP, (char *)&loopch, sizeof(loopch)) < 0)
  {
    vvDebugMsg::msg(2, "Setting IP_MULTICAST_LOOP error", true);
    return VV_SOCK_ERROR;
  }

  localInterface.s_addr = INADDR_ANY;
  if(setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_IF, (char *)&localInterface, sizeof(localInterface)) < 0)
  {
    vvDebugMsg::msg(2, "Setting local interface error", true);
    return VV_SOCK_ERROR;
  }

  return VV_OK;
}

//----------------------------------------------------------------------------
/// Initializes a multicast server (receiver)
vvSocket::ErrorType vvSocket::init_server_mc()
{
  vvDebugMsg::msg(3, "Enter vvSocket::init_server_mc()");

  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if(sockfd < 0)
  {
    vvDebugMsg::msg(1, "Error: socket()");
    return VV_SOCK_ERROR;
  }
  int reuse = 1;
  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse,sizeof(reuse)))
  {
    vvDebugMsg::msg(1, "Error: setsockopt()");
    return VV_SOCK_ERROR;
  }
  if (sock_buffsize > 0)
  {
    if(setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, (char *) &sock_buffsize, sizeof(sock_buffsize)) < 0)
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
    if(setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char *) &sock_buffsize, sizeof(sock_buffsize)) < 0)
    {
      vvDebugMsg::msg(1, "Error: setsockopt()");
      return VV_SOCK_ERROR;
    }
  }
  get_send_buffsize();
  ostringstream errmsg;
  errmsg << "send_buffsize: " << send_buffsize << " bytes, recv_buffsize: " << get_recv_buffsize() << " bytes";
  vvDebugMsg::msg(2, errmsg.str().c_str());
  if (send_buffsize < 65507)
    max_send_size = send_buffsize;
  else
    max_send_size = 65507;

  memset((char *) &localSock, 0, sizeof(localSock));
  localSock.sin_family = AF_INET;
  localSock.sin_port = htons(port);
  localSock.sin_addr.s_addr = INADDR_ANY;

  if(bind(sockfd, (struct sockaddr*)&localSock, sizeof(localSock)))
  {
    vvDebugMsg::msg(1, "Error: bind()");
    return VV_SOCK_ERROR ;
  }

  group.imr_multiaddr.s_addr = inet_addr(hostname);
  group.imr_interface.s_addr = INADDR_ANY;
  if(setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&group, sizeof(group)) < 0)
  {
    vvDebugMsg::msg(2, "Adding multicast group error", true);
    return VV_SOCK_ERROR;
  }

  return VV_OK;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
