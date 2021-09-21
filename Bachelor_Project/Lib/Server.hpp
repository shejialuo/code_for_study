/*
  * 实现服务器端的定义
*/
#ifndef SERVER_HPP
#define SERVER_HPP

#include <stdio.h>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string>
#include <unistd.h>
#define BUFFSIZE 100000

using std::string;

class Server {
private:
    int server_sockfd, client_sockfd;
    struct sockaddr_in serverAddr;
    struct sockaddr_in clientAddr;
    socklen_t addrLen;
    char buff[BUFFSIZE];
public:
    //构造函数
    Server(int port);
    //套接字
    int socket();
    //绑定
    int bind();
    //监听
    int listen(int queue = 5);
    //接收客户端的链接请求
    int accept();
    //接收客户端发送的消息
    string recv();
    //关闭套接字
    int close();
    //获取客户端的IP地址
    struct sockaddr_in getClientAddr();
};

#endif // SERVER_HPP