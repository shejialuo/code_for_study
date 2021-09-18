#ifndef TALOADBALANCER_HPP
#define TALOADBALANCER_HPP

#include <string>
#include <list>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"

using namespace std;

struct TextAnalysisInterface {
    string ipAddr;
};

class TextAnalyser_LoadBalancer {
private:
    list<TextAnalysisInterface> instancesConnected;
    int nextInstance;
    Server TALBServer;
public:
    TextAnalyser_LoadBalancer(int serverPort);
    void connectInstance(TextAnalysisInterface newInstance);
    TextAnalysisInterface disconnectInstance();
    void newRequest(string messageHeader, string messageBody, string messageId);
    void runServer();
};

#endif // TALOADBALANCER_HPP