#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <unordered_map>
#include <string>
#include "Data.hpp"
#include "../Lib/Server.hpp"
#include "../Lib/Client.hpp"

using namespace std;

class Database {
private:
    unordered_map<string, Results> actualMessage;
    int numberOfMessageInTheMonitoringWindow;
    unordered_map<string, string> startAnalysisTimes;
    double totalTime;
    int totalMessages;
    Server DBServer;
public:
    Database(int serverPort);
    void insertMessageInformation(string ip, string messageId, int linkNumber, int attachmentNumber);
    int insertHeadersAnalysisResults(HeaderAnalysis res, string ip);
    int insertLinksAnalysisResults(LinksAnalysis res, string ip);
    int insertTextAnalysisResults(TextAnalysis res, string ip);
    int insertAttachmentAnalysisResults(AttachmentAnalysis res, string ip);
    // Results getMessageResults(string messageId);
    int getNumberOfMessagesInTheMonitoringWindow();
    void resetNumberOfMessagesInTheMonitoringWindow();
    double getAverageAnalysisTime();
    void resetAverageAnalysisTime();
    void runServer();
};

#endif // DATABASE_HPP