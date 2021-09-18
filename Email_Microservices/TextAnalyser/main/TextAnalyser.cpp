#include "TextAnalyser.hpp"
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <future>
#include <vector>
#include <sstream>

class SentimentAsyncCall {
private:
    Server SentimentAsyncCallServer;
public:
    SentimentAsyncCall(int serverPort):SentimentAsyncCallServer(serverPort) {
        SentimentAsyncCallServer.socket();
        SentimentAsyncCallServer.bind();
        SentimentAsyncCallServer.listen(5000);
    }
    void newRequest(string messageBody) {
        Client newClient("192.168.90.110",8000,8001);
        newClient.socket();
        newClient.bind();
        newClient.connect();
        newClient.send(messageBody);
        newClient.close();
    }
    string operator()(string messageBody) {
        newRequest(messageBody);
        SentimentAsyncCallServer.accept();
        string results = SentimentAsyncCallServer.recv();
        return results;
    }
};

SentimentAsyncCall sentimentAsyncCall(8002);

TextAnalyser::TextAnalyser(int serverPort): TAServer(serverPort) {
    Client newClient("192.168.88.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string messagesInfo = "new connect";
    newClient.send(messagesInfo); 
    newClient.close();
}

void TextAnalyser::analyzeText(string messageHeader,
    string messageBody,string messageId) {
    sleep(2);
    if(!messageHeader.empty()) {
        
    }
    string resultOfSentiment;
    if(!messageBody.empty()) {
        auto asyncCall = std::async(sentimentAsyncCall, messageBody);
        resultOfSentiment = asyncCall.get();
    }
    Client newClient("192.168.102.110", 8000, 8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string results = "Texts " + messageBody + resultOfSentiment + " " + messageId; 
    newClient.send(results);
    newClient.close();
}

void TextAnalyser::runServer() {
    TAServer.socket();
    TAServer.bind();
    TAServer.listen(5000);
    while(true) {
        TAServer.accept();
        string messageInfo = TAServer.recv();
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
            string messageHeader = param.at(0);
            string messageBody = param.at(1);
            string messageId = param.at(2);
            analyzeText(messageHeader, messageBody, messageId);
        }
    }
}
