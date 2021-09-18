#include "NSFWDetector.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

NSFWDetector::NSFWDetector(int serverPort):NDServer(serverPort) {
    Client newClient("192.168.98.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string messagesInfo = "new connect";
    newClient.send(messagesInfo); 
    newClient.close();
    yesOrNot = -1;
}

void NSFWDetector::nsfwDetection(string image, string ip) {
    yesOrNot += 1;
    Client newClient(ip,8002,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    if (yesOrNot % 3 == 0) {
        newClient.send("false");
    }
    else {
        newClient.send("true");
    }
    newClient.close();
}

void NSFWDetector::runServer() {
    NDServer.socket();
    NDServer.bind();
    NDServer.listen(5000);
    while(true) {
        NDServer.accept();
        string messageInfo = NDServer.recv();
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
            string image = param.at(0);
            string ip = param.at(1);
            nsfwDetection(image, ip);
        }
    }
}