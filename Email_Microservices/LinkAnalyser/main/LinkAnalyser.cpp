#include "LinkAnalyser.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

LinkAnalyser::LinkAnalyser(int serverPort): LAServer(serverPort) {
   Client newClient("192.168.86.110",8000,8001);
   newClient.socket();
   newClient.bind();
   newClient.connect();
   string messagesInfo = "new connect";
   newClient.send(messagesInfo); 
   newClient.close();
}

void LinkAnalyser::analyzeLinks(string links, string messageId) {
   sleep(1);
   Client newClient("192.168.102.110",8000,8001);
   newClient.socket();
   newClient.bind();
   newClient.connect();
   string linkAnalyisisResult = "Links " + links + " " + messageId;
   newClient.send(linkAnalyisisResult);
   newClient.close();
}

void LinkAnalyser::runServer() {
   LAServer.socket();
   LAServer.bind();
   LAServer.listen(5000);
   while(true) {
      LAServer.accept();
      string messageInfo = LAServer.recv();
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
         string links = param.at(0);
         string messageId = param.at(1);
         analyzeLinks(links, messageId);
      }
   }
}