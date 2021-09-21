#include "MessageReceiver.hpp"

int main() {
    MessageReceiver messageReceiver(8000);
    messageReceiver.runServer();
    exit(0);
}