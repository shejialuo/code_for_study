#include "HeaderAnalyser.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

HeaderAnalyser::HeaderAnalyser(int serverPort): HAServer(serverPort) {
   Client newClient("192.168.84.110",8000,8001);
   newClient.socket();
   newClient.bind();
   newClient.connect();
   string messagesInfo = "new connect";
   newClient.send(messagesInfo); 
   newClient.close();
}

void HeaderAnalyser::analyzeHeaders(string headers, string messageId) {
   sleep(5);
   Client newClient("192.168.102.110",8000,8001);
   newClient.socket();
   newClient.bind();
   newClient.connect();
   string headerAnalyisisResult = "Headers " + headers + " " + messageId;
   newClient.send(headerAnalyisisResult);
   newClient.close();
}

void HeaderAnalyser::runServer() {
   HAServer.socket();
   HAServer.bind();
   HAServer.listen(5000);
   while(true) {
      HAServer.accept();
      string messageInfo = HAServer.recv();
      time_t now_time=time(NULL);  
      tm*  t_tm = localtime(&now_time);  
      string stt = asctime(t_tm);
      stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
      std::string logString = stt + " " + messageInfo;
      std::cout << logString << std::endl;;
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
         string headers = param.at(0);
         string messageId = param.at(1);
         analyzeHeaders(headers, messageId);
      }
   }
}