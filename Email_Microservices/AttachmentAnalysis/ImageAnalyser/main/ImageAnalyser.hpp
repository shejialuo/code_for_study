#ifndef IMAGEANALYSER_HPP
#define IMAGEANALYSER_HPP

#include <string>
#include "../../../Lib/Server.hpp"
#include "../../../Lib/Client.hpp"

using namespace std;

class ImageAnalyser {
private:
    Server IAServer;
public:
    ImageAnalyser(int serverPort);
    void analyzeImage(string image, string messageId);
    void runServer();
};

#endif // IMAGEANALYSER_HPP