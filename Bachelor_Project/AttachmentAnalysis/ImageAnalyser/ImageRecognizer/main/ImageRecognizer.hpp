#ifndef IMAGERECOGNIZER_HPP
#define IMAGERECOGNIZER_HPP

#include <string>
#include "../../../../Lib/Server.hpp"
#include "../../../../Lib/Client.hpp"

using namespace std;

class ImageRecognizer {
private:
    Server IRServer;
    int category;
public:
    ImageRecognizer(int serverPort);
    void recognizeImage(string image, string ip);
    void runServer();
};

#endif // IMAGERECOGNIZER_HPP