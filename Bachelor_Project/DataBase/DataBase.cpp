#include "DataBase.hpp"

#include <iostream>
#include <time.h>
#include <algorithm>
#include <vector>
#include <sstream>

Database::Database(int serverPort): DBServer(serverPort) {
    actualMessage = {};
    numberOfMessageInTheMonitoringWindow = 0;
    startAnalysisTimes = {};
    totalTime = 0;
    totalMessages = 0;
}

void Database::insertMessageInformation(string ip, string messageId, 
    int LinkNumber, int attachmentNumber) {

    int numberOfActivityWaiting = 2 + LinkNumber + attachmentNumber;
    vector<LinksAnalysis> linkAnalysis;
    vector<AttachmentAnalysis> attachmentAnalysis;
    Results newResults = Results(numberOfActivityWaiting,
                                  HeaderAnalysis("",""),
                                  linkAnalysis,
                                  TextAnalysis("",""),
                                  attachmentAnalysis);
    actualMessage.emplace(messageId,newResults);
    time_t now_time=time(NULL);  
    tm*  t_tm = localtime(&now_time);  
    string stt = asctime(t_tm);
    stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
    startAnalysisTimes.emplace(messageId, stt);
    ++numberOfMessageInTheMonitoringWindow;
    Client newClient(ip, 8002,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    newClient.send("OK");
    newClient.close();
}

int Database::insertHeadersAnalysisResults(HeaderAnalysis res, string ip) {
    int numberOfActivityWaiting = -1;
    actualMessage[res.haMessagedId].headerAnalysisResults.haMessagedId = res.haMessagedId;
    actualMessage[res.haMessagedId].headerAnalysisResults.haResults = res.haResults;
    actualMessage[res.haMessagedId].numberOfActivityWaiting - 1;
    numberOfActivityWaiting = actualMessage[res.haMessagedId].numberOfActivityWaiting;
    Client newCient(ip, 8003, 8001);
    newCient.socket();
    newCient.bind();
    newCient.connect();
    newCient.send(to_string(numberOfActivityWaiting));
    newCient.close();
    return numberOfActivityWaiting;
}

int Database::insertLinksAnalysisResults(LinksAnalysis res, string ip) {
    int numberOfActivityWaiting = -1;
    actualMessage[res.laMessageId].linksAnalysisResults.push_back(res);
    actualMessage[res.laMessageId].numberOfActivityWaiting - 1;
    numberOfActivityWaiting = actualMessage[res.laMessageId].numberOfActivityWaiting;
    Client newCient(ip, 8003, 8001);
    newCient.socket();
    newCient.bind();
    newCient.connect();
    newCient.send(to_string(numberOfActivityWaiting));
    newCient.close();
    return numberOfActivityWaiting;
}

int Database::insertTextAnalysisResults(TextAnalysis res, string ip) {
    int numberOfActivityWaiting = -1;
    actualMessage[res.laMessageId].textAnalysisResults.laMessageId = res.laMessageId;
    actualMessage[res.laMessageId].textAnalysisResults.laResults = res.laResults;
    actualMessage[res.laMessageId].numberOfActivityWaiting - 1;
    numberOfActivityWaiting = actualMessage[res.laMessageId].numberOfActivityWaiting;
    Client newCient(ip, 8003, 8001);
    newCient.socket();
    newCient.bind();
    newCient.connect();
    newCient.send(to_string(numberOfActivityWaiting));
    newCient.close();
    return numberOfActivityWaiting;
}

int Database::insertAttachmentAnalysisResults(AttachmentAnalysis res, string ip) {
    int numberOfActivityWaiting = -1;
    actualMessage[res.aaMessageId].attachmentAnalysisResults.push_back(res);
    actualMessage[res.aaMessageId].numberOfActivityWaiting - 1;
    numberOfActivityWaiting = actualMessage[res.aaMessageId].numberOfActivityWaiting;
    numberOfActivityWaiting = actualMessage[res.aaMessageId].numberOfActivityWaiting;
    Client newCient(ip, 8003, 8001);
    newCient.socket();
    newCient.bind();
    newCient.connect();
    newCient.send(to_string(numberOfActivityWaiting));
    newCient.close();
    return numberOfActivityWaiting;
}

int Database::getNumberOfMessagesInTheMonitoringWindow() {
    return numberOfMessageInTheMonitoringWindow;
}

void Database::resetNumberOfMessagesInTheMonitoringWindow() {
    numberOfMessageInTheMonitoringWindow = 0;
}

double Database::getAverageAnalysisTime() {
    double averageTime = 0;
    if (totalMessages != 0) {
        averageTime = totalTime / static_cast<double>(totalMessages);
    }
    return averageTime;
}

void Database::resetAverageAnalysisTime() {
    totalMessages = 0;
    totalTime = 0;
}

void Database::runServer() {
    DBServer.socket();
    DBServer.bind();
    DBServer.listen(5000);
    while(true) {
        DBServer.accept();
        string ip = inet_ntoa(DBServer.getClientAddr().sin_addr);
        string messageInfo = DBServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        std::cout << logString << std::endl;
        vector<string> param;
        istringstream iss(messageInfo);
        string token;
        while(getline(iss,token,' ')) {
            param.push_back(token);
        }
        token = param.at(0);
        if(strcmp(token.c_str(), "Headers") == 0) {
            string result = param.at(1) + " " + param.at(3);
            string messageId = param.at(2);
            HeaderAnalysis res = HeaderAnalysis(result, messageId);
            insertHeadersAnalysisResults(res, ip);
        }
        else if(strcmp(token.c_str(), "Links") == 0) {
            string result = param.at(1) + " " + param.at(3);
            string messageId = param.at(2);
            LinksAnalysis res = LinksAnalysis(result, messageId);
            insertLinksAnalysisResults(res, ip);
        }
        else if(strcmp(token.c_str(), "Texts") == 0) {
            string result = param.at(1) + " " + param.at(3);
            string messageId = param.at(2);
            TextAnalysis res = TextAnalysis(result, messageId);
            insertTextAnalysisResults(res, ip);
        }
        else if(strcmp(token.c_str(), "Attachments") == 0) {
            string result = param.at(1) + " " + param.at(3) + " " + param.at(4);
            string messageId = param.at(2);
            AttachmentAnalysis res =AttachmentAnalysis(result, messageId);
            insertAttachmentAnalysisResults(res, ip);
        }
        else {
            insertMessageInformation(ip,param.at(0),
             std::stoi(param.at(1)), std::stoi(param.at(2)));
        }
    }
}