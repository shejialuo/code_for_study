#include "TALoadBalancer.hpp"

int main() {
    TextAnalyser_LoadBalancer TALoader(8000);
    TALoader.runServer();
    exit(0);
}