#include "MessageParser.hpp"
#include <cstdlib>
#include <unordered_set>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <future>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

class DBInsert {
private:
    Server DBInsertServer;
public:
    DBInsert(int serverPort): DBInsertServer(serverPort) {
        DBInsertServer.socket();
        DBInsertServer.bind();
        DBInsertServer.listen(5000);
    }
    void newRequest(string request) {
        Client newClient("192.168.104.110", 8000, 8001);
        newClient.socket();
        newClient.bind();
        newClient.connect();
        newClient.send(request);
        newClient.close();
    }
    string operator()(string request) {
        newRequest(request);
        DBInsertServer.accept();
        string results = DBInsertServer.recv();
        return results;
    }
};

DBInsert DBInserter(8002);

MessageParser::MessageParser(int serverPort): MPServer(serverPort) {
    Client newClient("192.168.82.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string messagesInfo = "new connect";
    newClient.send(messagesInfo);
    newClient.close();
}

void MessageParser::headerAnalyserNewRequest(
    string header, string messageId) {
    Client newClient("192.168.84.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = header + " " + messageId;
    newClient.send(combinedString); 
    newClient.close();
}

void MessageParser::linkAnalyserNewRequest(
    string link, string messageId) {
    Client newClient("192.168.86.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = link + " " + messageId;
    newClient.send(combinedString); 
    newClient.close();
}

void MessageParser::textAnalyserNewqRequest(
    string messageHeader, string messageBody,
    string messageId) {
    Client newClient("192.168.88.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = messageHeader + " "  + messageBody + 
                            " " + messageId;
    newClient.send(combinedString); 
    newClient.close();
}

void MessageParser::attachmentAnalyserNewRequest(
    string attachment, string messageId) {
    Client newClient("192.168.92.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string combinedString = attachment + " " + messageId;
    newClient.send(combinedString); 
    newClient.close();
}

void MessageParser::parseMessage(string mailData) {
    string headers = mailData + "_NetworkHeaders";
    string sender = mailData + "_Sender";
    string messageHeader = mailData + "_MessageHeader";
    string messageBody = mailData + "_MessageBody";
    unordered_multiset<string> links;
    unordered_multiset<string> attachments;

    int numberOfLinks = rand() % 11;
    if(numberOfLinks > 0) {
        int i = 0;
        while (i < numberOfLinks) {
            string link = mailData + "_Link" + to_string(i);
            links.insert(link);
            ++i;
        }
    }
    else {
        links = {};
    }

    int numberOfAttachments = rand() % 5;
    if(numberOfAttachments > 0) {
        int i = 0;
        while(i < numberOfAttachments) {
            string attachment = mailData + "_Attachment" + to_string(i);
            attachments.insert(attachment);
            ++i;
        }
    }
    else {
        attachments = {};
    }

    MessageFields messageFields = {
        headers, sender, messageHeader, 
        messageBody, links, attachments
    };

    boost::uuids::uuid newUuid = boost::uuids::random_generator()();
    string messageId = boost::uuids::to_string(newUuid);
    

    string res = messageId + " " + std::to_string(numberOfLinks) + " "
                 + std::to_string(numberOfAttachments);
    auto futureDNInsert = std::async(DBInserter, res);
    headerAnalyserNewRequest(messageFields.headers, messageId);

    if(!links.empty()) {
        for(auto &linkx: messageFields.links) {
            linkAnalyserNewRequest(linkx, messageId);
        }
        links.clear();
    }

    textAnalyserNewqRequest(messageFields.messageHeader,
                            messageFields.messageBody ,messageId);

    if(!attachments.empty()) {
        for(auto &attachment:messageFields.attachments) {
            attachmentAnalyserNewRequest(attachment,messageId);
        }
        attachments.clear();
    }
}

void MessageParser::runServer() {
    MPServer.socket();
    MPServer.bind();
    MPServer.listen(5000);
    while(true) {
        MPServer.accept();
        string messageInfo = MPServer.recv();
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
            parseMessage(messageInfo);
        }
    }
}