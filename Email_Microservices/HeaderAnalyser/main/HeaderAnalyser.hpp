#ifndef HEADERANALYSER_HPP
#define HEADERANALYSER_HPP

#include <string>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"
using namespace std;

class HeaderAnalyser {
private:
    Server HAServer;
public:
    HeaderAnalyser(int serverPort);
    void analyzeHeaders(string headers, string messageId);
    void runServer();
};

#endif // HEADERANALYSER_HPP