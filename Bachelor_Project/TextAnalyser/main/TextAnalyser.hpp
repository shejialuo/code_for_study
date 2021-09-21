#ifndef TEXTANALYSER_HPP
#define TEXTANALYSER_HPP

#include <string>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"

using namespace std;

class TextAnalyser {
private:
    Server TAServer;
public:
    TextAnalyser(int serverPort);
    void analyzeText(string messageHeader, string messageBody,
                     string messageId);
    void runServer();
};

#endif // TEXTANALYSER_HPP