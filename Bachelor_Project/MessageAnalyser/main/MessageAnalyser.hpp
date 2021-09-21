#ifndef MESSAGEANALYSER_HPP
#define MESSAGEANALYSER_HPP

#include <string>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"

using namespace std;

class MessageAnalyser {
private:
    Server MAServer;
public:
    MessageAnalyser(int serverPort);
    void insertHeadersAnalysisResults(string res);
    void insertLinksAnalysisResults(string res);
    void insertTextAnalysisResults(string res);
    void insertAttachmentAnalysisResults(string res);
    void runServer();
};

#endif // MESSAGEANALYSER_HPP