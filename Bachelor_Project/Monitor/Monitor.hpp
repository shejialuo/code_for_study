#ifndef MONITOR_HPP
#define MONITOR_HPP

#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <future>
#include <thread>
#include <mutex>
#include "../Lib/Client.hpp"

using namespace std;

class Monitor {
private:
    int monitorWindowDimension;
    int numberOfHedaerAnalyser;
    int numberOfLinkAnalyser;
    int numberOfTextAnalyser;
    int numberOfSentimentAnalyser;
    int numberOfImageAnalyser;
    int numberOfImageRecgonizer;
    int numberOfNSFWDetector;
    int messageCounter;
    int index;
    vector<int> numberOfMessagesInAMonitorWindow;
public:
    Monitor();
    string stringSplit(string& message, int n, char splitChar);
    int stringFindNumber(string& message);
    string shellExecute(string& command);
    string dataBaseQuery(string& command);
    vector<string> headerAnalyserQuery();
    vector<string> linkAnalyserQuery();
    vector<string> textAnalyserQuery();
    vector<string> sentimentAnalyserQuery();
    void scaleHeaderAnalyser(string& s1, vector<string>& s2);
    void scaleLinkAnalyser(string& s1, vector<string>& s2);
    void scaleTextAnalyser(string& s1, vector<string>& s2);
    void scaleSentimentAnalyser(string& s1, vector<string>& s2);
    void run();
};


#endif // MONITOR_HPP