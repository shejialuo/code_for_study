#include "MALoadBalancer.hpp"
#include <iterator>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <time.h>
#include <algorithm>

MessageAnalyser_LoadBalancer::MessageAnalyser_LoadBalancer(int serverPort)
    :MALBServer(serverPort){
    instancesConnected = {};
    nextInstance = -1;
}

void MessageAnalyser_LoadBalancer::connectInstance(
    MessageAnalyserInterface newInstance) {
    instancesConnected.push_back(newInstance);
}

MessageAnalyserInterface MessageAnalyser_LoadBalancer::
    disconnectInstance() {
    MessageAnalyserInterface removedInstance = instancesConnected.back();
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

void MessageAnalyser_LoadBalancer::insertHeadersAnalysisResults(string res) {
    nextInstance = (nextInstance + 1) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    MessageAnalyserInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    newClient.send(res);
    newClient.close();
}

void MessageAnalyser_LoadBalancer::insertLinksAnalysisResults(string res) {
    nextInstance = (nextInstance + 1) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    MessageAnalyserInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    newClient.send(res);
    newClient.close();
}

void MessageAnalyser_LoadBalancer::insertTextAnalysisResults(string res) {
    nextInstance = (nextInstance + 1) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    MessageAnalyserInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    newClient.send(res);
    newClient.close();
}

void MessageAnalyser_LoadBalancer::insertAttachmentAnalysisResults(string res) {
    nextInstance = (nextInstance + 1) % instancesConnected.size();
    auto pos = instancesConnected.cbegin();
    advance(pos, nextInstance);
    MessageAnalyserInterface selectedInstance = *pos;
    Client newClient(selectedInstance.ipAddr, 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    newClient.send(res);
    newClient.close();
}

void MessageAnalyser_LoadBalancer::runServer() {
    MALBServer.socket();
    MALBServer.bind();
    MALBServer.listen(5000);
    std::ofstream ofile;
    ofile.open("./log.txt", ios::app);
    while(true) {
        MALBServer.accept();
        string messageInfo = MALBServer.recv();
        string ip = inet_ntoa(MALBServer.getClientAddr().sin_addr);
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        ofile << logString << std::endl;
        if(strcmp(messageInfo.c_str(), "new connect") == 0) {
            MessageAnalyserInterface newInstance {
                inet_ntoa(MALBServer.getClientAddr().sin_addr)};
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
            token = param.at(0);
            string res = messageInfo + " " + ip;
            if(strcmp(token.c_str(), "Headers") == 0) {
                insertHeadersAnalysisResults(res);
            }
            else if(strcmp(token.c_str(), "Links") == 0) {
                insertLinksAnalysisResults(res);
            }
            else if(strcmp(token.c_str(), "Texts") == 0) {
                insertTextAnalysisResults(res);
            }
            else {
                insertAttachmentAnalysisResults(res);
            }
        }
    }
}