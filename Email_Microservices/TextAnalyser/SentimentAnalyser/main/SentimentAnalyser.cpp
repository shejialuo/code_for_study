#include "SentimentAnalyser.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

SentimentAnalyser::SentimentAnalyser(int serverPort): SAServer(serverPort) {
   Client newClient("192.168.90.110",8000,8001);
   newClient.socket();
   newClient.bind();
   newClient.connect();
   string messagesInfo = "new connect";
   newClient.send(messagesInfo); 
   newClient.close();
}

void SentimentAnalyser::analyzeSentiment(string messageBody, string ip) {
    sleep(4);
    Client newClient(ip,8002,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    newClient.send("SentimentAnalysis");
    newClient.close();
}

void SentimentAnalyser::runServer() {
    SAServer.socket();
    SAServer.bind();
    SAServer.listen(5000);
    while(true) {
        SAServer.accept();
        string messageInfo = SAServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;
        if(strcmp(messageInfo.c_str(), "disconnect") == 0) {
         exit(0);   
        }
        else {
            vector<string> param;
            istringstream iss(messageInfo);
            string token;
            while(getline(iss,token,' ')) {
                param.push_back(token);
            }
            string messageBody = param.at(0);
            string ip = param.at(1);
            analyzeSentiment(messageBody, ip);
        }
    }
}