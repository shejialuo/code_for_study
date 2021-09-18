#include "ImageAnalyser.hpp"

#include <iostream>
#include <vector>
#include <sstream>
#include <future>
#include <thread>
#include <time.h>
#include <algorithm>

class NSFWAsyncCall {
private:
    Server NSFWAsyncCallServer;
public:
    NSFWAsyncCall(int serverPort): NSFWAsyncCallServer(serverPort) {
        NSFWAsyncCallServer.socket();
        NSFWAsyncCallServer.bind();
        NSFWAsyncCallServer.listen(5000);
    }
    void newRequest(string image) {
        Client newClient("192.168.98.110",8000,8001);
        newClient.socket();
        newClient.bind();
        newClient.connect();
        newClient.send(image);
        newClient.close();
    }
    string operator()(string image) {
        newRequest(image);
        NSFWAsyncCallServer.accept();
        string results = NSFWAsyncCallServer.recv();
        return results;
    }
};

class ImageRecognizerAsyncCall {
private:
    Server IRAsyncCallServer;
public:
    ImageRecognizerAsyncCall(int serverPort): IRAsyncCallServer(serverPort) {
        IRAsyncCallServer.socket();
        IRAsyncCallServer.bind();
        IRAsyncCallServer.listen(5000);
    }
    void newRequest(string image) {
        Client newClient("192.168.100.110",8000,8001);
        newClient.socket();
        newClient.bind();
        newClient.connect();
        newClient.send(image);
        newClient.close();
    }
    string operator()(string image) {
        newRequest(image);
        IRAsyncCallServer.accept();
        string results = IRAsyncCallServer.recv();
        return results;
    }
};

NSFWAsyncCall nsfwAsyncCall(8002);
ImageRecognizerAsyncCall IRAsyncCall(8003);

ImageAnalyser::ImageAnalyser(int serverPort): IAServer(serverPort) {
    Client newClient("192.168.96.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string messagesInfo = "new connect";
    newClient.send(messagesInfo); 
    newClient.close();
}

void ImageAnalyser::analyzeImage(string image, string messageId) {
    auto futureNSFW = std::async(nsfwAsyncCall, image);
    auto futureIR = std::async(IRAsyncCall, image);
    string NSFWResult = futureNSFW.get();
    string IRResult = futureIR.get();
    string result;
    if(strcmp(NSFWResult.c_str(), "true") == 0) {
        result = "Attachments " + image + " " + messageId + " NSFW";
    }
    else {
        result = "Attachments " + image + " " + messageId + " " + IRResult;
    }
    Client newClient("192.168.102.110", 8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    newClient.send(result);
    newClient.close();
}

void ImageAnalyser::runServer() {
    IAServer.socket();
    IAServer.bind();
    IAServer.listen(5000);
    while(true) {
      IAServer.accept();
      string messageInfo = IAServer.recv();
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
         string messageId = param.at(1);
         analyzeImage(image, messageId);
      }
   }
}