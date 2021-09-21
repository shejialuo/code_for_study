#include "AttachmentManager.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <time.h>
#include <algorithm>

AttachmentManager::AttachmentManager(int serverPort): AMServer(serverPort) {
    Client newClient("192.168.94.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string messagesInfo = "new connect";
    newClient.send(messagesInfo); 
    newClient.close();
}

void AttachmentManager::manageAttachments(string attachment, string messageId) {
    Client newClient("192.168.96.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combined = attachment + " " + messageId;
    newClient.send(combined); 
    newClient.close();  
}

void AttachmentManager::runServer() {
    AMServer.socket();
    AMServer.bind();
    AMServer.listen(5000);
    while(true) {
      AMServer.accept();
      string messageInfo = AMServer.recv();
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
         string attachment = param.at(0);
         string messageId = param.at(1);
         manageAttachments(attachment, messageId);
      }
   }
}