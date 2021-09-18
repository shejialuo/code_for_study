#include "Server.hpp"

Server::Server(int port) {
    addrLen = sizeof(struct sockaddr_in);
    memset(&serverAddr,0,sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr=htons(INADDR_ANY);
    serverAddr.sin_port = htons(port);
}

int Server::socket() {
    server_sockfd = ::socket(AF_INET,SOCK_STREAM,0);
    return server_sockfd;
}

int Server::bind() {
    return ::bind(server_sockfd,(struct sockaddr*)
                &serverAddr,sizeof(struct sockaddr));
}

int Server::listen(int queue) {
    return ::listen(server_sockfd, queue);
}

int Server::accept() {
    client_sockfd = ::accept(server_sockfd,(struct sockaddr*)
                    &clientAddr,&addrLen);
    return client_sockfd;
}

string Server::recv() {
    int len = ::recv(client_sockfd, buff, BUFSIZ, 0);
    buff[len] = '\0';
    string response = buff;
    memset(buff, '\0', sizeof(buff));
    return response;
}

int Server::close() {
    return ::close(server_sockfd);
}

struct sockaddr_in Server::getClientAddr() {
    return clientAddr;
}