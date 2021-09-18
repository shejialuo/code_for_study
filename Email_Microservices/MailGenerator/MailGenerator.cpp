#include "MailGenerator.hpp"
#include <iostream>
#include <unistd.h>
#include <time.h>
#include <algorithm>
using namespace std;

MailGenerator::MailGenerator() {
    messageCounter = 0;
    // numberOfMessagesRequiredInAMonitorWindow = {
    //     10, 10, 10, 10, 10, 10, 10,
	// 	30, 30, 30, 30, 30, 30, 30,
	// 	80, 80, 80, 80, 80, 80, 80,
	// 	160, 160, 160, 160, 160, 160, 160,
	// 	180, 180, 180, 180, 180, 180, 180,
	// 	100, 100, 100, 100, 100, 100, 100,
	// 	80, 80, 80, 80, 80, 80, 80,
	// 	60, 60, 60, 60, 60, 60, 60
    // };
    numberOfMessagesRequiredInAMonitorWindow = {
        30,50,80,30,50,20,50,80,10
    };
    monitoringWindowDimension = 200;
}

void MailGenerator::run() {
    int i = 0;
    while(true) {
        int currentNumberOfMessageRequired = numberOfMessagesRequiredInAMonitorWindow.at(i);
        i = (i + 1) % numberOfMessagesRequiredInAMonitorWindow.size();
        double timeBetweenTwoConsecutiveMessages = 
            static_cast<double>(monitoringWindowDimension) /static_cast<double>(currentNumberOfMessageRequired);

        int j = 0;
        while(j < currentNumberOfMessageRequired) {
            string mailData = "Message" + to_string(messageCounter);

            time_t now_time=time(NULL);  
            tm*  t_tm = localtime(&now_time);  
            string stt = asctime(t_tm);
            stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
            std::string logString = stt + " " + mailData;
            std::cout << logString << std::endl;

            Client newClient("192.168.81.110", 8000, 8001);
            newClient.socket();
            newClient.bind();
            newClient.connect();
            newClient.send(mailData);
            newClient.close();

            messageCounter = messageCounter + 1;
            sleep(timeBetweenTwoConsecutiveMessages);
            j++;
        }
    }
}