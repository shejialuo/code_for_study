#include "SentimentAnalyser.hpp"

int main() {
    SentimentAnalyser sentimentAnalyser(8000);
    sentimentAnalyser.runServer();
    exit(0);
}