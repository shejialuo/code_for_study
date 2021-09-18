#include "ImageRecognizer.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>

ImageRecognizer::ImageRecognizer(int serverPort):IRServer(serverPort) {
    Client newClient("192.168.100.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string messagesInfo = "new connect";
    newClient.send(messagesInfo); 
    newClient.close();
    category = -1;
}

void ImageRecognizer::recognizeImage(string image, string ip) {
    category = category + 1;
    Client newClient(ip,8003,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string results = "Category" + to_string(category);
    newClient.send(results);
    newClient.close();
}

void ImageRecognizer::runServer() {
    IRServer.socket();
    IRServer.bind();
    IRServer.listen(5000);
    while(true) {
        IRServer.accept();
        string messageInfo = IRServer.recv();
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
            recognizeImage(image, ip);
        }
    }
}