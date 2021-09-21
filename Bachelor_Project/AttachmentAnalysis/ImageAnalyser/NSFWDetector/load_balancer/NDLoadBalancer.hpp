#ifndef NDLOADBALANCER_HPP
#define NDLOADBALANCER_HPP

#include <string>
#include <list>
#include "../../../../Lib/Server.hpp"
#include "../../../../Lib/Client.hpp"

using namespace std;

struct NSFWInterface {
    string ipAddr;
};

class NSFWDetector_LoadBalancer {
private:
    list<NSFWInterface> instancesConnected;
    int nextInstance;
    Server NSFWLBServer;
public:
    NSFWDetector_LoadBalancer(int serverPort);
    void connectInstance(NSFWInterface newInstance);
    NSFWInterface disconnectInstance();
    void newRequest(string image, string ip);
    void runServer();
};

#endif // NDLOADBALANCER_HPP