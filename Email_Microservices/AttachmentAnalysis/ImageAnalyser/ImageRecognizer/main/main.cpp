#include "ImageRecognizer.hpp"

int main() {
    ImageRecognizer imageRecognizer(8000);
    imageRecognizer.runServer();
    exit(0);
}