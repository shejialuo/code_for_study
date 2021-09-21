#include "HALoadBalancer.hpp"

int main() {
    HeaderAnalyser_LoadBalancer HALoader(8000);
    HALoader.runServer();
    exit(0);
}