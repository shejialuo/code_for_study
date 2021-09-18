#include "MessageReceiver.hpp"
#include <iostream>
#include <time.h>
#include <algorithm>

MessageReceiver::MessageReceiver(int port): MRServer(port) {}

void MessageReceiver::newMessage(string mailData) {
    Client newClient("192.168.82.110", 8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    newClient.send(mailData);
    newClient.close();
}

void MessageReceiver::runServer() {
    MRServer.socket();
    MRServer.bind();
    MRServer.listen(5000);
    while(true) {
        MRServer.accept();
        std::string messageInfo = MRServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;
        newMessage(messageInfo);
    }
}