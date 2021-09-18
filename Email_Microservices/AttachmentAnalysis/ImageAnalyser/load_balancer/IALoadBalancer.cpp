#include "IALoadBalancer.hpp"
#include <iterator>
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

ImageAnalyser_LoadBalancer::ImageAnalyser_LoadBalancer(
    int serverPort): IALBServer(serverPort) {
    instancesConnected = {};
    nextInstance = -1;
}

void ImageAnalyser_LoadBalancer::connectInstance(
    ImageAnalysisInterface newInstance) {
    instancesConnected.push_back(newInstance);
}

ImageAnalysisInterface ImageAnalyser_LoadBalancer
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

void ImageAnalyser_LoadBalancer::newRequest(
    string image, string messageId) {
    nextInstance =  (nextInstance + 1 ) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    ImageAnalysisInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = image + " " + messageId;
    newClient.send(combinedString);
    newClient.close();
}

void ImageAnalyser_LoadBalancer::runServer() {
    IALBServer.socket();
    IALBServer.bind();
    IALBServer.listen(5000);
    while(true) {
        IALBServer.accept();
        string messageInfo = IALBServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;

        if(strcmp(messageInfo.c_str(), "new connect") == 0) {
            ImageAnalysisInterface newInstance {
                inet_ntoa(IALBServer.getClientAddr().sin_addr)};
            connectInstance(newInstance);
        }
        else if(strcmp(messageInfo.c_str(), "disconnect") == 0) {
            disconnectInstance();
        }
        else {
            vector<string> param;
            istringstream iss(messageInfo);
            string token;
            while(getline(iss,token,' ')) {
                param.push_back(token);
            }
            string image = param.at(0);
            string messageId = param.at(1);
            newRequest(image, messageId);
        }
    }
}