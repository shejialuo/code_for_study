#ifndef VIRUSSCANNER_HPP
#define VIRUSSCANNER_HPP

#include <string>
#include "../../../Lib/Server.hpp"
#include "../../../Lib/Client.hpp"

using namespace std;

class VirusScanner {
private:
    Server VSServer;
public:
    VirusScanner(int serverPort);
    void scanAttachment(string headers, string messageId);
    void runServer();
};

#endif // VIRUSSCANNER_HPP