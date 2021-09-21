#include "NDLoadBalancer.hpp"

int main() {
    NSFWDetector_LoadBalancer NSFWLoader(8000);
    NSFWLoader.runServer();
    exit(0);
}