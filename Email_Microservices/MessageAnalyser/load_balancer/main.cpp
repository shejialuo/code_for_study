#include "MALoadBalancer.hpp"

int main() {
    MessageAnalyser_LoadBalancer MALoader(8000);
    MALoader.runServer();
    exit(0);
}