#include "Monitor.hpp"

int main() {
    Monitor monitor;
    string command = "/home/shejialuo/Projects/Email_Microservices/scripts/depolyment.sh";
    monitor.shellExecute(command);
    monitor.run();
    exit(0);
}