#include "DataBase.hpp"

int main() {
    Database database(8000);
    database.runServer();
    exit(0);
}