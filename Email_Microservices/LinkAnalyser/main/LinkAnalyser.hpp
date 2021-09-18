#ifndef LINKANALYSER_HPP
#define LINKANALYSER_HPP

#include <string>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"
using namespace std;

class LinkAnalyser {
private:
    Server LAServer;
public:
    LinkAnalyser(int serverPort);
    void analyzeLinks(string links, string messageId);
    void runServer();
};

#endif // LINKANALYSER_HPP