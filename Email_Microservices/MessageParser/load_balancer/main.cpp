#include "MPLoadBalancer.hpp"

int main() {
    MessageParser_LoadBalancer MPLoader(8000);
    MPLoader.runServer();
    exit(0);
}