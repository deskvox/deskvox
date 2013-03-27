// Virvo - Virtual Reality Volume Rendering
// Contact: Stefan Zellmann, zellmans@uni-koeln.de
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

#ifndef VV_LIBRARY_BUILD
#error "vvpthread.h is meant for internal use only"
#endif

#ifndef _VV_PTHREAD_H_
#define _VV_PTHREAD_H_

#include <pthread.h>

/* Pthread barriers aren't available on Mac OS X 10.3.
 * Albeit we know that there are other Unixes that don't implement
 * barriers either, we only use our barrier implementation when
 * compiling on Mac OS X. The #ifdef below  might be changed to a
 * more reasonable value if needed.
 */
#ifdef __APPLE__
#define VV_USE_CUSTOM_BARRIER_IMPLEMENTATION
typedef struct
{
  int count;
  int waited;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} barrier_t;

typedef struct
{
  int dummy;
} barrierattr_t;

#define pthread_barrier_t barrier_t
#define pthread_barrierattr_t barrierattr_t
#define pthread_barrier_init(b, a, n) barrier_init(b, a, n)
#define pthread_barrier_destroy(b) barrier_destroy(b)
#define pthread_barrier_wait(b) barrier_wait(b)

int pthread_barrier_init(pthread_barrier_t* barrier,
                         const pthread_barrierattr_t* attr,
                         unsigned int count);
int pthread_barrier_destroy(pthread_barrier_t* barrier);
int pthread_barrier_wait(pthread_barrier_t* barrier);
#endif

namespace virvo
{

//------------------------------------------------------------------------------
// Mutex
//
class Mutex
{
  pthread_mutex_t mutex;

public:
  Mutex() { pthread_mutex_init(&mutex, 0); }
 ~Mutex() { pthread_mutex_destroy(&mutex); }

  // Returns a pointer to the internal pthread_mutex_t
  pthread_mutex_t const* get() const { return &mutex; }

  // Returns a pointer to the internal pthread_mutex_t
  pthread_mutex_t* get() { return &mutex; }

  // Lock
  void lock() { pthread_mutex_lock(get()); }

  // Unlock
  void unlock() { pthread_mutex_unlock(get()); }

  // Try to lock.
  // Returns true on success, false otherwise.
  bool tryLock() { return pthread_mutex_trylock(get()) == 0; }
};

//------------------------------------------------------------------------------
// ScopedLock
//
// Locks the given mutex upon construction and unlocks
// the mutex when destructed.
//
class ScopedLock
{
  Mutex* mutex;

public:
  ScopedLock(Mutex* mutex) : mutex(mutex) { mutex->lock(); }
 ~ScopedLock() { mutex->unlock(); }
};

//------------------------------------------------------------------------------
// Condition
//
class Condition
{
  pthread_cond_t cond;

public:
  Condition() { pthread_cond_init(&cond, 0); }
 ~Condition() { pthread_cond_destroy(&cond); }

  // Returns a pointer to the internal pthread_cond_t
  pthread_cond_t const* get() const { return &cond; }

  // Returns a pointer to the internal pthread_cond_t
  pthread_cond_t* get() { return &cond; }

  // Calling thread waits for the condition
  // NOTE: Mutex must be locked!
  void wait(Mutex* mutex) { pthread_cond_wait(get(), mutex->get()); }

  // Wake up any thread waiting on the condition
  void signal() { pthread_cond_signal(get()); }

  // Wake up all threads waiting on the condition
  void broadcast() { pthread_cond_broadcast(get()); }
};

//------------------------------------------------------------------------------
// SyncedCondition
//
// Same as Condition, but includes a mutex to synchronize the condition
// variable.
//
class SyncedCondition
{
  // Mutex to sync the condition variable
  Mutex mutex;
  // The actual condition variable
  Condition cond;

public:
  // Calling thread blocks until the condition variable becomes signaled
  void wait()
  {
    ScopedLock lock(&mutex);
    cond.wait(&mutex);
  }

  // Wake up any thread waiting on the condition
  void signal() { cond.signal(); }

  // Wake up all threads waiting on the condition
  void broadcast() { cond.broadcast(); }
};

} // namespace virvo

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
