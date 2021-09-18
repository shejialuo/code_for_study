/*
  * 定义客户端
*/
#ifndef CLIENT_HPP
#define CLIENT_HPP

#include <stdio.h>
#include <unistd.h>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string>
using std::string;

#define BUFFSIZE 100000

class Client {
private:
    int sockfd;
    struct sockaddr_in serverAddr;
    struct sockaddr_in clientAddr;
    socklen_t addrLen;
    char buff[BUFFSIZE];
public:
    //构造函数
    Client(string serverIp, int serverPort, int clientPort);
    //创建socket
    int socket();
    //绑定
    int bind();
    //与服务器端建立链接
    int connect();
    //向服务器端发送信息
    int send(string messageInfo);
    //关闭socket
    int close();
};

#endif // CLIENT_HPP