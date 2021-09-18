#ifndef IRLOADBALANCER_HPP
#define IRLOADBALANCER_HPP

#include <string>
#include <list>
#include "../../../../Lib/Server.hpp"
#include "../../../../Lib/Client.hpp"

using namespace std;

struct ImageAnalysisInterface {
    string ipAddr;
};

class ImageRecognizer_LoadBalancer {
private:
    list<ImageAnalysisInterface> instancesConnected;
    int nextInstance;
    Server IRLBServer;
public:
    ImageRecognizer_LoadBalancer(int serverPort);
    void connectInstance(ImageAnalysisInterface newInstance);
    ImageAnalysisInterface disconnectInstance();
    void newRequest(string image, string ip);
    void runServer();
};

#endif // IRLOADBALANCER_HP