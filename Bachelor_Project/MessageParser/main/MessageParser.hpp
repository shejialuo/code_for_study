#ifndef MESSAGEPARSER_HPP
#define MESSAGEPARSER_HPP

#include <string>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"
#include "../../Data/MessageField.hpp"
using namespace std;

class MessageParser {
private:
    Server MPServer;
public:
    MessageParser(int serverPort);
    void headerAnalyserNewRequest(string headers, string messageId);
    void linkAnalyserNewRequest(string links, string messageId);
    void textAnalyserNewqRequest(string messageHeader, string messageBody, string messageId);
    void attachmentAnalyserNewRequest(string attachment, string messageId);
    void parseMessage(string mailData);
    void runServer();
};

#endif // MESSAGEPARSER_HPP