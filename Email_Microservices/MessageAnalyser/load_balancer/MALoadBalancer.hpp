#ifndef MALOADBALANCER_HPP
#define MALOADBALANCER_HPP

#include <string>
#include <list>
#include "../../Lib/Server.hpp"
#include "../../Lib/Client.hpp"

using namespace std;

struct MessageAnalyserInterface {
    string ipAddr;
};

class MessageAnalyser_LoadBalancer {
private:
    list<MessageAnalyserInterface> instancesConnected;
    int nextInstance;
    Server MALBServer;
public:
    MessageAnalyser_LoadBalancer(int serverPort);
    void connectInstance(MessageAnalyserInterface newInstance);
    MessageAnalyserInterface disconnectInstance();
    void insertHeadersAnalysisResults(string res);
	void insertLinksAnalysisResults(string res);
	void insertTextAnalysisResults(string res);
	void insertAttachmentAnalysisResults(string res);
    void runServer();
};

#endif // MALOADBALANCER_HPP