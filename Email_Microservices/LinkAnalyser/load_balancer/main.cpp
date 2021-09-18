#include "LALoadBalancer.hpp"

int main() {
    LinkAnalyser_LoadBalancer LALoader(8000);
    LALoader.runServer();
    exit(0);
}