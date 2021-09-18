#ifndef SALOADBALANCER_HPP
#define SALOADBALANCER_HPP

#include <string>
#include <list>
#include "../../../Lib/Server.hpp"
#include "../../../Lib/Client.hpp"

using namespace std;

struct SentimentAnalysisInterface {
    string ipAddr;
};

class SentimentAnalyser_LoadBalancer {
private:
    list<SentimentAnalysisInterface> instancesConnected;
    int nextInstance;
    Server SALBServer;
public:
    SentimentAnalyser_LoadBalancer(int serverPort);
    void connectInstance(SentimentAnalysisInterface newInstance);
    SentimentAnalysisInterface disconnectInstance();
    void newRequest(string messageBody, string ip);
    void runServer();
};

#endif // SALOADBALANCER_HPP