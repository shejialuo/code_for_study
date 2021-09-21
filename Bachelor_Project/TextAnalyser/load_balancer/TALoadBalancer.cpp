#include "TALoadBalancer.hpp"
#include <vector>
#include <iterator>
#include <iostream>
#include <sstream>
#include <time.h>
#include <algorithm>

TextAnalyser_LoadBalancer::TextAnalyser_LoadBalancer(
    int serverPort): TALBServer(serverPort){
    instancesConnected = {};
    nextInstance = -1;    
}

TextAnalysisInterface TextAnalyser_LoadBalancer
    ::disconnectInstance() {
    TextAnalysisInterface removedInstance = instancesConnected.back();
    instancesConnected.pop_back();
    return removedInstance;
}

void TextAnalyser_LoadBalancer::connectInstance(
    TextAnalysisInterface newInstance) {
    instancesConnected.push_back(newInstance);
}

void TextAnalyser_LoadBalancer::newRequest(
    string messageHeader, string messageBody, string messageId) {
    nextInstance =  (nextInstance + 1) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    TextAnalysisInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = messageHeader + " " + messageBody
                            + " " + messageId;
    newClient.send(combinedString);
    newClient.close();
}

void TextAnalyser_LoadBalancer::runServer() {
    TALBServer.socket();
    TALBServer.bind();
    TALBServer.listen(5000);
    while(true) {
        TALBServer.accept();
        string messageInfo = TALBServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;
        if(strcmp(messageInfo.c_str(), "new connect") == 0) {
            TextAnalysisInterface newInstance {
                inet_ntoa(TALBServer.getClientAddr().sin_addr)};
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
            string messageHeader = param.at(0);
            string messageBody = param.at(1);
            string messageId = param.at(2);
            newRequest(messageHeader, messageBody, messageId);
        }
    }
}