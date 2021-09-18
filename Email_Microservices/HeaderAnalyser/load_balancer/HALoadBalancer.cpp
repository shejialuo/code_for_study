#include "HALoadBalancer.hpp"
#include <iterator>
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

HeaderAnalyser_LoadBalancer::HeaderAnalyser_LoadBalancer(
    int serverPort): HALBServer(serverPort) {
    instancesConnected = {};
    nextInstance = -1;
}

void HeaderAnalyser_LoadBalancer::connectInstance(
    HeaderAnalysisInterface newInstance) {
    instancesConnected.push_back(newInstance);
}

HeaderAnalysisInterface HeaderAnalyser_LoadBalancer
    ::disconnectInstance() {
    HeaderAnalysisInterface removedInstance = instancesConnected.back();
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

void HeaderAnalyser_LoadBalancer::newRequest(
    string headers, string messageId) {
    nextInstance =  (nextInstance + 1 ) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    HeaderAnalysisInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = headers + " " + messageId;
    newClient.send(combinedString);
    newClient.close();
}

void HeaderAnalyser_LoadBalancer::runServer() {
    HALBServer.socket();
    HALBServer.bind();
    HALBServer.listen(5000);
    while(true) {
        HALBServer.accept();
        string messageInfo = HALBServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;
        if(strcmp(messageInfo.c_str(), "new connect") == 0) {
            HeaderAnalysisInterface newInstance {
                inet_ntoa(HALBServer.getClientAddr().sin_addr)};
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
            string headers = param.at(0);
            string messageId = param.at(1);
            newRequest(headers, messageId);
        }
    }
}