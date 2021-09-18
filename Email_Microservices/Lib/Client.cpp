#include "Client.hpp"

Client::Client(string serverIp, int serverPort, int clientPort) {
    addrLen = sizeof(struct sockaddr_in);
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr(serverIp.c_str());
    serverAddr.sin_port = htons(serverPort);
    clientAddr.sin_family = AF_INET;
    clientAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    clientAddr.sin_port = htonl(clientPort);
}

int Client::socket() {
    sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
    return sockfd;
}

int Client::bind() {
    return ::bind(sockfd,(struct sockaddr*)
                  &clientAddr,sizeof(struct sockaddr));
}

int Client::connect() {
    return ::connect(sockfd, (struct sockaddr *)&serverAddr, addrLen);
}

int Client::send(string messageInfo) {
    for(int i = 0; i < messageInfo.length(); i++) {
        buff[i] = messageInfo[i];
    }
    int result = ::send(sockfd, buff, strlen(buff), 0);
    memset(buff, '\0', sizeof(buff));
    return result;
}

int Client::close() {
    return ::close(sockfd);
}
