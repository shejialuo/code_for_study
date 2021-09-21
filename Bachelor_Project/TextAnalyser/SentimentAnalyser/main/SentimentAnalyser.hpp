#ifndef SENTIMENTANALYSER_HPP
#define SENTIMENTANALYSER_HPP

#include <string>
#include "../../../Lib/Client.hpp"
#include "../../../Lib/Server.hpp"

using namespace std;

class SentimentAnalyser {
private:
    Server SAServer;
public:
    SentimentAnalyser(int serverPort);
    void analyzeSentiment(string messageBody, string ip);
    void runServer();
};

#endif // TEXTANALYSER_HPP