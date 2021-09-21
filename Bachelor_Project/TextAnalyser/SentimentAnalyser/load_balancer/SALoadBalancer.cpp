#include "SALoadBalancer.hpp"
#include <vector>
#include <iterator>
#include <iostream>
#include <sstream>
#include <time.h>
#include <algorithm>

SentimentAnalyser_LoadBalancer::SentimentAnalyser_LoadBalancer(
    int serverPort): SALBServer(serverPort){
    instancesConnected = {};
    nextInstance = -1;    
}

SentimentAnalysisInterface SentimentAnalyser_LoadBalancer
    ::disconnectInstance() {
    SentimentAnalysisInterface removedInstance = instancesConnected.back();
    instancesConnected.pop_back();
    return removedInstance;
}

void SentimentAnalyser_LoadBalancer::connectInstance(
    SentimentAnalysisInterface newInstance) {
    instancesConnected.push_back(newInstance);
}

void SentimentAnalyser_LoadBalancer::newRequest(string messageBody, string ip) {
    nextInstance =  (nextInstance + 1) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    SentimentAnalysisInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combined = messageBody + " " + ip;
    newClient.send(combined);
    newClient.close();
}

void SentimentAnalyser_LoadBalancer::runServer() {
    SALBServer.socket();
    SALBServer.bind();
    SALBServer.listen(5000);
    while(true) {
        SALBServer.accept();
        string ip = inet_ntoa(SALBServer.getClientAddr().sin_addr);
        string messageInfo = SALBServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;;
        if(strcmp(messageInfo.c_str(), "new connect") == 0) {
            SentimentAnalysisInterface newInstance {
                inet_ntoa(SALBServer.getClientAddr().sin_addr)};
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