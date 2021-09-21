#ifndef MPLOADBALANCER_HPP
#define MPLOADBALANCER_HPP

#include <string>
#include <list>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"

using namespace std;

struct MessageParserInterface {
    string ipAddr;
};

class MessageParser_LoadBalancer {
private:
    list<MessageParserInterface> instancesConnected;
    int nextInstance;
    Server MPLBServer;
public:
    MessageParser_LoadBalancer(int serverPort);
    void connectInstance(MessageParserInterface newInstance);
    MessageParserInterface disconnectInstance();
    void newRequest(string mailData);
    void runServer();
};

#endif // MPLOADBALANCER_HPP