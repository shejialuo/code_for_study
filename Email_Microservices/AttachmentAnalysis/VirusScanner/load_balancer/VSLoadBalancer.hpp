#ifndef VSLOADBALANCER_HPP
#define VSLOADBALANCER_HPP

#include <string>
#include <list>
#include "../../../Lib/Server.hpp"
#include "../../../Lib/Client.hpp"

using namespace std;

struct VirusScannerInterface {
    string ipAddr;
};

class VirusScanner_LoadBalancer {
private:
    list<VirusScannerInterface> instancesConnected;
    int nextInstance;
    Server VSLBServer;
public:
    VirusScanner_LoadBalancer(int serverPort);
    void connectInstance(VirusScannerInterface newInstance);
    VirusScannerInterface disconnectInstance();
    void newRequest(string attachment, string messageId);
    void runServer();
};

#endif // VSLOADBALANCER_HPP