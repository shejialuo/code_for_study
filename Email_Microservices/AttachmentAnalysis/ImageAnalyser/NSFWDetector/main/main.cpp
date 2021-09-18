#include "NSFWDetector.hpp"

int main() {
    NSFWDetector NSFWdetector(8000);
    NSFWdetector.runServer();
    exit(0);
}