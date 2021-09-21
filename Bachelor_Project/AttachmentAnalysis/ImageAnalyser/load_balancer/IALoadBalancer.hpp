#ifndef IALOADBALANCER_HPP
#define IALOADBALANCER_HPP

#include <list>
#include <string>
#include "../../../Lib/Server.hpp"
#include "../../../Lib/Client.hpp"
using namespace std;

struct ImageAnalysisInterface {
    string ipAddr;
};

class ImageAnalyser_LoadBalancer {
private:
    list<ImageAnalysisInterface> instancesConnected;
    int nextInstance;
    Server IALBServer;
public:
    ImageAnalyser_LoadBalancer(int serverPort);
    void connectInstance(ImageAnalysisInterface newInstance);
    ImageAnalysisInterface disconnectInstance();
    void newRequest(string image, string messageId);
    void runServer();
};

#endif // IALOADBALANCER_HPP