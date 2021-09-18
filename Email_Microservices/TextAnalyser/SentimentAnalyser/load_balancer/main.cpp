#include "SALoadBalancer.hpp"

int main() {
    SentimentAnalyser_LoadBalancer SALoader(8000);
    SALoader.runServer();
    exit(0);
}