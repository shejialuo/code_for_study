#ifndef MAILRECEIVER_HPP
#define MAILRECEIVER_HPP

#include "../Lib/Server.hpp"
#include "../Lib/Client.hpp"

using namespace std;

class MessageReceiver {
private:
    Server MRServer;
public:
    MessageReceiver(int port);
    void newMessage(string mailData);
    void runServer();
};

#endif // MAILRECEIVER_HPP