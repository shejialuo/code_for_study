#ifndef ATTACHMENTMANAGER_HPP
#define ATTACHMENTMANAGER_HPP

#include <string>
#include "../../../Lib/Server.hpp"
#include "../../../Lib/Client.hpp"

using namespace std;

class AttachmentManager {
private:
    Server AMServer;
public:
    AttachmentManager(int serverPort);
    void manageAttachments(string attachment, string messageId);
    void runServer();
};

#endif // ATTACHMENTMANAGER_HPP