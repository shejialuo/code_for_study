#include "IRLoadBalancer.hpp"

int main() {
    ImageRecognizer_LoadBalancer IRLoader(8000);
    IRLoader.runServer();
    exit(0);
}