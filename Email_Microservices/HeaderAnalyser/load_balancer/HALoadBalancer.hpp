#ifndef HALOADBALANCER_HPP
#define HALOADBALANCER_HPP

#include <string>
#include <list>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"

using namespace std;

struct HeaderAnalysisInterface {
    string ipAddr;
};

class HeaderAnalyser_LoadBalancer {
private:
    list<HeaderAnalysisInterface> instancesConnected;
    int nextInstance;
    Server HALBServer;
public:
    HeaderAnalyser_LoadBalancer(int serverPort);
    void connectInstance(HeaderAnalysisInterface newInstance);
    HeaderAnalysisInterface disconnectInstance();
    void newRequest(string headers, string messageId);
    void runServer();
};

#endif // HALOADBALANCER_HPP