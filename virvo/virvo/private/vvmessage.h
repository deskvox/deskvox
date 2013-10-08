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


#ifndef VV_PRIVATE_MESSAGE_H
#define VV_PRIVATE_MESSAGE_H


#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>

#include <boost/smart_ptr/enable_shared_from_this.hpp>
#include <boost/smart_ptr/make_shared.hpp>

#include <boost/function.hpp>

#include <cassert>
#include <stdexcept>
#include <vector>


namespace virvo
{


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------


    class Message
        : public boost::enable_shared_from_this<Message>
    {
        friend class Client;
        friend class Server;

        static unsigned GenerateID()
        {
            static unsigned counter = 0;
            return ++counter;
        }

        typedef std::vector<char> data_type;
        typedef data_type::value_type element_type;

        struct Header
        {
            // The unique ID of this message
            unsigned id_;
            // The type of this message
            unsigned type_;
            // The length of this message
            unsigned size_;

            Header(unsigned id, unsigned type, unsigned size)
                : id_(id)
                , type_(type)
                , size_(size)
            {
            }
        };

        // The message data
        data_type data_;
        // The message header
        Header header_;

    public:
        explicit Message(unsigned type = 0)
            : data_()
            , header_(GenerateID(), type, 0)
        {
        }

        template<class InputIterator>
        explicit Message(unsigned type, InputIterator first, InputIterator last)
            : data_(first, last)
            , header_(GenerateID(), type, static_cast<unsigned>(data_.size()))
        {
        }

        // Creates a serialized message
        template<class T>
        explicit Message(unsigned type, T const& object)
            : data_()
            , header_(GenerateID(), type, 0)
        {
            typedef boost::iostreams::back_insert_device<data_type> sink_type;
            typedef boost::iostreams::stream<sink_type> stream_type;

            sink_type sink(data_);
            stream_type stream(sink);

            {
                // Create a serializer
                boost::archive::binary_oarchive archive(stream);

                // Serialize the message
                archive << object;

                // Don't forget to flush the stream!!!
                stream.flush();
            }
            //~archive

            // Set the size of the serialized message
            header_.size_ = static_cast<unsigned>(data_.size());
        }

        // Deserialize the message
        template<class T>
        bool deserialize(T& object) const
        {
            assert( header_.size_ == data_.size() );

            typedef boost::iostreams::basic_array_source<element_type> source_type;
            typedef boost::iostreams::stream<source_type> stream_type;

            source_type source(&data_[0], data_.size());
            stream_type stream(source);

            // Create a deserialzer
            boost::archive::binary_iarchive archive(stream);

            // Deserialize the message
            archive >> object;

            return static_cast<bool>(stream);
        }

        template<class T>
        T deserialize() const
        {
            T object;

            if (!deserialize(object))
                throw std::runtime_error("deserialization error");

            return object;
        }

        // Returns the unique ID of this message
        unsigned id() const {
            return header_.id_;
        }

        // Returns the type of this message
        unsigned type() const {
            return header_.type_;
        }

        // Returns the size of the message
        unsigned size() const
        {
            assert( header_.size_ == data_.size() );
            return static_cast<unsigned>(data_.size());
        }

        // Returns an iterator to the first element of the data
        data_type::iterator begin() {
            return data_.begin();
        }

        // Returns an iterator to the element following the last element of the data
        data_type::iterator end() {
            return data_.end();
        }

        // Returns an iterator to the first element of the data
        data_type::const_iterator begin() const {
            return data_.begin();
        }

        // Returns an iterator to the element following the last element of the data
        data_type::const_iterator end() const {
            return data_.end();
        }
    };


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------


    typedef boost::shared_ptr<Message> MessagePointer;

    inline MessagePointer makeMessage(unsigned type = 0)
    {
        return boost::make_shared<Message>(type);
    }

    template<class InputIterator>
    MessagePointer makeMessage(unsigned type, InputIterator first, InputIterator last)
    {
        return boost::make_shared<Message>(type, first, last);
    }

    template<class T>
    MessagePointer makeMessage(unsigned type, T const& object)
    {
        return boost::make_shared<Message>(type, object);
    }


} // namespace virvo


#endif // !VV_PRIVATE_MESSAGE_H
