#include "VSLoadBalancer.hpp"
#include <iterator>
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

VirusScanner_LoadBalancer::VirusScanner_LoadBalancer(
    int serverPort): VSLBServer(serverPort) {
    instancesConnected = {};
    nextInstance = -1;
}

void VirusScanner_LoadBalancer::connectInstance(
    VirusScannerInterface newInstance) {
    instancesConnected.push_back(newInstance);
}

VirusScannerInterface VirusScanner_LoadBalancer
    ::disconnectInstance() {
    VirusScannerInterface removedInstance = instancesConnected.back();
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

void VirusScanner_LoadBalancer::newRequest(
    string attachment, string messageId) {
    nextInstance =  (nextInstance + 1 ) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    VirusScannerInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = attachment + " " + messageId;
    newClient.send(combinedString);
    newClient.close();
}

void VirusScanner_LoadBalancer::runServer() {
    VSLBServer.socket();
    VSLBServer.bind();
    VSLBServer.listen(5000);
    while(true) {
        VSLBServer.accept();
        string messageInfo = VSLBServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;

        if(strcmp(messageInfo.c_str(), "new connect") == 0) {
            VirusScannerInterface newInstance {
                inet_ntoa(VSLBServer.getClientAddr().sin_addr)};
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
            string attachment = param.at(0);
            string messageId = param.at(1);
            newRequest(attachment, messageId);
        }
    }
}