#ifndef AMLOADBALANCER_HPP
#define AMLOADBALANCER_HPP

#include <string>
#include <list>
#include "../../../Lib/Server.hpp"
#include "../../../Lib/Client.hpp"

using namespace std;

struct AttachmentManagerInterface {
    string ipAddr;
};

class AttachmentManager_LoadBalancer {
private:
    list<AttachmentManagerInterface> instancesConnected;
    int nextInstance;
    Server AMLBServer;
public:
    AttachmentManager_LoadBalancer(int serverPort);
    void connectInstance(AttachmentManagerInterface newInstance);
    AttachmentManagerInterface disconnectInstance();
    void newRequest(string attachment, string messageId);
    void runServer();
};

#endif // AMLOADBALANCER_HPP