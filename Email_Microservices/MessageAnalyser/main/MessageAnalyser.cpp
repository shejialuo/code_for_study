#include "MessageAnalyser.hpp"

#include <iostream>
#include <vector>
#include <sstream>
#include <future>
#include <thread>
#include <fstream>
#include <time.h>
#include <algorithm>

class DBAnalysis {
private:
    Server DBServer;
public:
    DBAnalysis(int serverPort): DBServer(serverPort) {
        DBServer.socket();
        DBServer.bind();
        DBServer.listen(5000);
    }
    void newRequest(string res) {
        Client newClient("192.168.104.110",8000,8001);
        newClient.socket();
        newClient.bind();
        newClient.connect();
        newClient.send(res);
        newClient.close();
    }
    string operator()(string res) {
        newRequest(res);
        DBServer.accept();
        string numberOfActivityWaiting = DBServer.recv();
        return numberOfActivityWaiting;
    }
};

DBAnalysis DBAsyncCall(8003);

MessageAnalyser::MessageAnalyser(int serverPort):MAServer(serverPort) {
    Client newClient("192.168.102.110",8000,8001);
    newClient.socket();
    newClient.bind();
    newClient.connect();
    string messagesInfo = "new connect";
    newClient.send(messagesInfo); 
    newClient.close(); 
}

void MessageAnalyser::insertHeadersAnalysisResults(string res) {

    auto futureResults = std::async(DBAsyncCall, res);
    int numberOfActivityWaiting = std::stoi(futureResults.get());
    if(numberOfActivityWaiting == 0) {
        std::cout << "All done" << std::endl;
    }
}

void MessageAnalyser::insertLinksAnalysisResults(string res) {
    auto futureResults = std::async(DBAsyncCall, res);
    int numberOfActivityWaiting = std::stoi(futureResults.get());
    if(numberOfActivityWaiting == 0) {
        std::cout << "All done" << std::endl;
    }
}

void MessageAnalyser::insertTextAnalysisResults(string res) {
    auto futureResults = std::async(DBAsyncCall, res);
    int numberOfActivityWaiting = std::stoi(futureResults.get());
    if(numberOfActivityWaiting == 0) {
        std::cout << "All done" << std::endl;
    }
}

void MessageAnalyser::insertAttachmentAnalysisResults(string res) {
    auto futureResults = std::async(DBAsyncCall, res);
    int numberOfActivityWaiting = std::stoi(futureResults.get());
    if(numberOfActivityWaiting == 0) {
        std::cout << "All done" << std::endl;
    }
}

void MessageAnalyser::runServer() {
    MAServer.socket();
    MAServer.bind();
    MAServer.listen(5000);
    std::ofstream ofile;
    ofile.open("./log.txt", ios::app);
    while(true) {
        MAServer.accept();
        string messageInfo = MAServer.recv();
        time_t now_time=time(NULL);  
        tm*  t_tm = localtime(&now_time);  
        string stt = asctime(t_tm);
        stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
        std::string logString = stt + " " + messageInfo;
        ofile << logString << std::endl;
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
            token = param.at(0);
            if(strcmp(token.c_str(), "Headers") == 0) {
                insertHeadersAnalysisResults(messageInfo);
            }
            else if(strcmp(token.c_str(), "Links") == 0) {
                insertLinksAnalysisResults(messageInfo);
            }
            else if(strcmp(token.c_str(), "Texts")) {
                insertTextAnalysisResults(messageInfo);
            }
            else {
                insertAttachmentAnalysisResults(messageInfo);
            }
        }
    }
}