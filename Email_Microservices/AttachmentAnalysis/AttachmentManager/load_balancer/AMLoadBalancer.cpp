#include "AMLoadBalancer.hpp"
#include <iterator>
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

AttachmentManager_LoadBalancer::AttachmentManager_LoadBalancer(
    int serverPort): AMLBServer(serverPort) {
    instancesConnected = {};
    nextInstance = -1;
}

void AttachmentManager_LoadBalancer::connectInstance(
    AttachmentManagerInterface newInstance) {
    instancesConnected.push_back(newInstance);
}

AttachmentManagerInterface AttachmentManager_LoadBalancer
    ::disconnectInstance() {
    AttachmentManagerInterface removedInstance = instancesConnected.back();
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

void AttachmentManager_LoadBalancer::newRequest(
    string attachment, string messageId) {
    nextInstance =  (nextInstance + 1 ) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    AttachmentManagerInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = attachment + " " + messageId;
    newClient.send(combinedString);
    newClient.close();
}

void AttachmentManager_LoadBalancer::runServer() {
    AMLBServer.socket();
    AMLBServer.bind();
    AMLBServer.listen(5000);
    while(true) {
        AMLBServer.accept();
        string messageInfo = AMLBServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;

        if(strcmp(messageInfo.c_str(), "new connect") == 0) {
            AttachmentManagerInterface newInstance {
                inet_ntoa(AMLBServer.getClientAddr().sin_addr)};
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