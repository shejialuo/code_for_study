#ifndef MAILGENERATOR_HPP
#define MAILGENERATOR_HPP

#include "../Lib/Client.hpp"
#include <vector>

using namespace std;

class MailGenerator {
private:
    int messageCounter;
    vector<int> numberOfMessagesRequiredInAMonitorWindow;
    int monitoringWindowDimension;
public:
    MailGenerator();
    void run();
};

#endif // MAILGENERATOR_HPP