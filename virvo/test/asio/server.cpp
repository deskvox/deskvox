// server.cpp


#include "vvserver.h"

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <cstdio>
#include <iostream>
#include <iomanip>


using namespace virvo;


namespace boost { namespace uuids {

    std::ostream& operator <<(std::ostream& stream, uuid const& id)
    {
        for (size_t I = 0; I < 16; ++I)
            stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(id.data[I]);

        return stream;
    }

}}


class MyServer : public Server
{
public:
    MyServer()
    {
    }

    //--- Server interface -----------------------------------------------------

    virtual void on_read(ServerManager::ConnectionPointer conn, MessagePointer message)
    {
        std::cout << "SERVER Message read: " << message->id() << " : \"" << message->deserialize<std::string>() << "\"" << std::endl;
#if 0
        static_cast<void>(conn);
#else
        conn->write(message);
#endif
    }

    virtual void on_write(ServerManager::ConnectionPointer /*conn*/, MessagePointer message)
    {
        std::cout << "SERVER Message sent: " << message->id() << std::endl;
    }
};


class MyServerManager : public ServerManager
{
public:
    MyServerManager()
        : ServerManager(30000)
    {
    }

    //--- ServerManager interface ----------------------------------------------

    virtual bool on_accept(ServerManager::ConnectionPointer conn, boost::system::error_code const& /*e*/)
    {
        std::cout << "MyServerManager: new connection accepted" << std::endl;

        // Wrap a new MyServer in conn
        return conn->accept(boost::make_shared<MyServer>());
    }
};


int main()
{
    try
    {
        // Create a new server manager
        MyServerManager mgr;

        // Start reading/writing messages
        mgr.run();
    }
    catch (std::exception& e)
    {
        std::cerr << "SERVER EXCEPTION: " << e.what() << std::endl;
    }
}
