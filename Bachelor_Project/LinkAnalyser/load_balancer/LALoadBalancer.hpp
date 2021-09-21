#ifndef LALOADBALANCER_HPP
#define LALOADBALANCER_HPP

#include <string>
#include <list>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"

using namespace std;

struct LinkAnalysisInterface {
    string ipAddr;
};

class LinkAnalyser_LoadBalancer {
private:
    list<LinkAnalysisInterface> instancesConnected;
    int nextInstance;
    Server LALBServer;
public:
    LinkAnalyser_LoadBalancer(int serverPort);
    void connectInstance(LinkAnalysisInterface newInstance);
    LinkAnalysisInterface disconnectInstance();
    void newRequest(string links, string messageId);
    void runServer();
};

#endif // HALOADBALANCER_HPP