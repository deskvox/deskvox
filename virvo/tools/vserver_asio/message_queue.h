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

#ifndef VSERVER_MESSAGE_QUEUE_H
#define VSERVER_MESSAGE_QUEUE_H

#include <virvo/private/vvmessage.h>

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

#ifndef NDEBUG
#include <iostream>
#endif
#include <list>

// Thread safe (blocking) queue to store messages
class vvMessageQueue
{
    typedef boost::mutex Mutex;
    typedef boost::unique_lock<Mutex> LockGuard;
    typedef boost::condition_variable ConditionVariable;

    // The queue
    std::list<virvo::MessagePointer> queue_;
    // A mutex to protect the queue.
    Mutex lock_;
    // Signaled if the queue becomes non-empty
    ConditionVariable non_empty_;

public:
    // Clear the message queue
    void clear()
    {
        LockGuard guard(lock_);

        queue_.clear();
    }

    // Wake up the first waiting thread - if any
    void notify_one()
    {
        non_empty_.notify_one();
    }

    // Unconditionally push a message into the queue
    void push_back_always(virvo::MessagePointer message)
    {
        {
            LockGuard guard(lock_);

            // Add the message to the message queue
            queue_.push_back(message);
        }

        notify_one();
    }

    // Only push_back the message if the last message is of a different type.
    // E.g., allows to merge some frame requests.
    void push_back(virvo::MessagePointer message)
    {
        {
            LockGuard guard(lock_);

            if (queue_.empty() || queue_.back()->type() != message->type())
            {
                // Add the message to the message queue
                queue_.push_back(message);
            }
            else
            {
                // Replace the existing message with the new message
                queue_.back() = message;
            }
        }

        notify_one();
    }

    // Returns the front of the message queue.
    // If the queue is empty, the calling thread waits until the queue
    // becomes non-empty.
    void pop_front_blocking(virvo::MessagePointer& message)
    {
        LockGuard guard(lock_);

        //while (queue_.empty())
        {
            non_empty_.wait(guard);
        }

        // Get the next message
        message = queue_.front();
        // Remove the message from the queue
        queue_.pop_front();
    }
};

#endif
