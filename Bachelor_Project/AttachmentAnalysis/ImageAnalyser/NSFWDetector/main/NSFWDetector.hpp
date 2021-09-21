#ifndef NSFWDETECTOR_HPP
#define NSFWDETECTOR_HPP

#include <string>
#include "../../../../Lib/Server.hpp"
#include "../../../../Lib/Client.hpp"

using namespace std;

class NSFWDetector {
private:
    Server NDServer;
    int yesOrNot;
public:
    NSFWDetector(int serverPort);
    void nsfwDetection(string image, string ip);
    void runServer();
};

#endif // NSFWDETECTOR_HPP