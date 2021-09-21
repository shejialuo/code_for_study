#include "IRLoadBalancer.hpp"
#include <iterator>
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

ImageRecognizer_LoadBalancer::ImageRecognizer_LoadBalancer(
    int serverPort): IRLBServer(serverPort) {
    instancesConnected = {};
    nextInstance = -1;
}

void ImageRecognizer_LoadBalancer::connectInstance(
    ImageAnalysisInterface newInstance) {
    instancesConnected.push_back(newInstance);
}

ImageAnalysisInterface ImageRecognizer_LoadBalancer
    ::disconnectInstance() {
    ImageAnalysisInterface removedInstance = instancesConnected.back();
    instancesConnected.pop_back();
    Client newClient(removedInstance.ipAddr, 8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string messageInfo = "disconnect";
    newClient.send(messageInfo);
    newClient.close();
    return removedInstance;
}

void ImageRecognizer_LoadBalancer::newRequest(
    string image, string ip) {
    nextInstance =  (nextInstance + 1 ) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    ImageAnalysisInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = image + " " + ip;
    newClient.send(combinedString);
    newClient.close();
}

void ImageRecognizer_LoadBalancer::runServer() {
    IRLBServer.socket();
    IRLBServer.bind();
    IRLBServer.listen(5000);
    while(true) {
        IRLBServer.accept();
        string ip = inet_ntoa(IRLBServer.getClientAddr().sin_addr);
        string messageInfo = IRLBServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;
        
        if(strcmp(messageInfo.c_str(), "new connect") == 0) {
            ImageAnalysisInterface newInstance {
                inet_ntoa(IRLBServer.getClientAddr().sin_addr)};
            connectInstance(newInstance);
        }
        else if(strcmp(messageInfo.c_str(), "disconnect") == 0) {
            disconnectInstance();
        }
        else {
            newRequest(messageInfo, ip);
        }
    }
}