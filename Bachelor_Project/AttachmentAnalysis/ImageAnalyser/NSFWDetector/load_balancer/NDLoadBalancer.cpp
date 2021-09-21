#include "NDLoadBalancer.hpp"
#include <iterator>
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

NSFWDetector_LoadBalancer::NSFWDetector_LoadBalancer(
    int serverPort): NSFWLBServer(serverPort) {
    instancesConnected = {};
    nextInstance = -1;
}

void NSFWDetector_LoadBalancer::connectInstance(
    NSFWInterface newInstance) {
    instancesConnected.push_back(newInstance);
}

NSFWInterface NSFWDetector_LoadBalancer
    ::disconnectInstance() {
    NSFWInterface removedInstance = instancesConnected.back();
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

void NSFWDetector_LoadBalancer::newRequest(
    string image, string ip) {
    nextInstance =  (nextInstance + 1 ) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    NSFWInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = image + " " + ip;
    newClient.send(combinedString);
    newClient.close();
}

void NSFWDetector_LoadBalancer::runServer() {
    NSFWLBServer.socket();
    NSFWLBServer.bind();
    NSFWLBServer.listen(5000);
    while(true) {
        NSFWLBServer.accept();
        string ip = inet_ntoa(NSFWLBServer.getClientAddr().sin_addr);
        string messageInfo = NSFWLBServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;;
        
        if(strcmp(messageInfo.c_str(), "new connect") == 0) {
            NSFWInterface newInstance {
                inet_ntoa(NSFWLBServer.getClientAddr().sin_addr)};
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