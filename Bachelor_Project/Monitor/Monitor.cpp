#include "Monitor.hpp"

mutex io_mutex;

string Monitor::stringSplit(string& message, int n, char splitChar) {
    vector<string> param;
    istringstream iss(message);
    string token;
    while(getline(iss, token, splitChar)) {
        param.push_back(token);
    }
    return param.at(n);
}

int Monitor::stringFindNumber(string& message) {
    int length = 0;
    int number = 0;
    for(auto &m: message) {
        if(m >= '0' && m <= '9') {
            length++;
        }
    }
    for(auto &m: message) {
        if(m >= '0' && m <= '9') {
            number+= (m - '0') * pow(10, length - 1);
            length--;
        }
    }
    return number;
}

string Monitor::shellExecute(string& command) {
    FILE *fp = NULL;
	char *buff = NULL;
	buff = (char*)malloc(500);
	memset(buff, 0, 500);
	fp = popen(command.c_str(), "r");
	fgets(buff, 500, fp);
	string result = buff;
	pclose(fp);
	free(buff);
    return result;
}

Monitor::Monitor() {
    messageCounter = -1;
    // numberOfMessagesInAMonitorWindow = {
    //     10, 10, 10, 10, 10, 
	// 	80, 80, 80, 80, 80, 
	// 	160, 160, 160, 160, 160, 
	// 	180, 180, 180, 180, 180, 
	// 	100, 100, 100, 100, 100, 
	// 	80, 80, 80, 80, 80, 
	// 	60, 60, 60, 60, 60, 
    // };
    numberOfMessagesInAMonitorWindow = {
        30,50,80,30,50,20,50,80,10
    };
    monitorWindowDimension = 200;
    numberOfHedaerAnalyser = 1;
    numberOfLinkAnalyser = 1;
    numberOfTextAnalyser = 1;
    numberOfSentimentAnalyser = 1;
    numberOfImageAnalyser = 1;
    numberOfImageRecgonizer = 1;
    numberOfNSFWDetector = 1;
    index = 0;
}

string Monitor::dataBaseQuery(string& command) {
    return shellExecute(command);
}

vector<string> Monitor::headerAnalyserQuery() {
    vector<string> results;
    for(int i = 1; i <= numberOfHedaerAnalyser; i++) {
            string command = "sudo docker logs headeranalyser"+ to_string(i) + " | tail -1";
            string shellResult = shellExecute(command);
            results.push_back(shellResult);
    }
    return results;
}

vector<string> Monitor::linkAnalyserQuery() {
    vector<string> results;
    for(int i = 1; i <= numberOfLinkAnalyser; i++) {
            string command = "sudo docker logs linkanalyser"+ to_string(i) + " | tail -1";
            string shellResult = shellExecute(command);
            results.push_back(shellResult);
    }
    return results;
}

vector<string> Monitor::textAnalyserQuery() {
    vector<string> results;
    for(int i = 1; i <= numberOfTextAnalyser; i++) {
            string command = "sudo docker logs textanalyser"+ to_string(i) + " | tail -1";
            string shellResult = shellExecute(command);
            results.push_back(shellResult);
    }
    return results;
}

vector<string> Monitor::sentimentAnalyserQuery() {
    vector<string> results;
    for(int i = 1; i <= numberOfSentimentAnalyser; i++) {
            string command = "sudo docker logs sentimentanalyser"+ to_string(i) + " | tail -1";
            string shellResult = shellExecute(command);
            results.push_back(shellResult);
    }
    return results;
}

void Monitor::scaleHeaderAnalyser(string& dataBaseQuery, vector<string>& headerAnalyserQuery) {
    string resultSplited = stringSplit(dataBaseQuery, 7,' ');
    int arriveNumber = stringFindNumber(resultSplited);
    int difference = messageCounter - arriveNumber;
    io_mutex.lock();
    cout << "目前未到达的HeaderAnalyserResults数目为：" << difference << endl;
    io_mutex.unlock();
    if( difference > 0) {
        int numberOfHeaderAnalyserInAMonitorTime = monitorWindowDimension / 5;
        double scaleUpNumber_ = static_cast<double>(difference) / 
                                static_cast<double>(numberOfHeaderAnalyserInAMonitorTime);
        int scaleUpNumber = ceil(scaleUpNumber_);
        for(int i = 1; i <= scaleUpNumber; i++) {
            string command = "/home/shejialuo/Projects/Email_Microservices/scripts/HeaderAnalyser.sh headeranalyser"
                             + to_string(i + numberOfHedaerAnalyser);
            shellExecute(command);
            time_t now_time=time(NULL);  
            tm*  t_tm = localtime(&now_time);  
            string stt = asctime(t_tm);
            stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
            std::string logString = stt + " " + "创建了HeaderAnalyser" + to_string(i + numberOfHedaerAnalyser);
            io_mutex.lock();
            std::cout << logString << std::endl;
            io_mutex.unlock();
        }
        numberOfHedaerAnalyser += scaleUpNumber;
    }
    else {
        vector<int> headerAnalyserQuery_;
        for(auto &query: headerAnalyserQuery) {
            string stringSplited = stringSplit(query, 6, ' ');
            headerAnalyserQuery_.push_back(stringFindNumber(stringSplited));
        }
        sort(headerAnalyserQuery_.begin(),headerAnalyserQuery_.end());
        bool flag = true;
        int n = headerAnalyserQuery_.at(headerAnalyserQuery_.size() -1 ) -headerAnalyserQuery_.at(0);
        if (n > 40) {
            flag = false;
            int scaleNumber = n / 40;
            for(int i = 1; i <= scaleNumber; ++i) {
                string command = "/home/shejialuo/Projects/Email_Microservices/scripts/HeaderAnalyser.sh headeranalyser"
                                + to_string(i + numberOfHedaerAnalyser);
                shellExecute(command);
                time_t now_time=time(NULL);  
                tm*  t_tm = localtime(&now_time);  
                string stt = asctime(t_tm);
                stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
                std::string logString = stt + " " + "创建了HeaderAnalyser" + to_string(i + numberOfHedaerAnalyser);
                io_mutex.lock();
                std::cout << logString << std::endl;
                io_mutex.unlock();
            }
            numberOfHedaerAnalyser += scaleNumber;
        }
        if(!flag) return; 
        int messageNumber = numberOfMessagesInAMonitorWindow.at(index);
        double numberRequired_ = static_cast<double>(messageNumber) / 
                                static_cast<double>(monitorWindowDimension / 5);
        int numberRequired = ceil(numberRequired_);
        while(numberRequired < numberOfHedaerAnalyser) {
            Client newClient("192.168.84.110",8000,8001);
            newClient.socket();
            newClient.bind();
            newClient.connect();
            newClient.send("disconnect");
            newClient.close();
            string command = "sudo docker container rm -f headeranalyser" + to_string(numberOfHedaerAnalyser);
            shellExecute(command);
            time_t now_time=time(NULL);  
            tm*  t_tm = localtime(&now_time);  
            string stt = asctime(t_tm);
            stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
            std::string logString = stt + " " + "删除了HeaderAnalyser" + to_string(numberOfHedaerAnalyser);
            io_mutex.lock();
            std::cout << logString << std::endl;
            io_mutex.unlock();
            numberOfHedaerAnalyser--;
        }
    }
}

void Monitor::scaleTextAnalyser(string& saQuery, vector<string>& textAnalyserQuery) {
    string resultSplited = stringSplit(saQuery, 6,' ');
    int arriveNumber = stringFindNumber(resultSplited);
    int difference = messageCounter - arriveNumber;
    io_mutex.lock();
    cout << "目前未到达的TextAnalyserResults数目为：" << difference << endl;
    io_mutex.unlock();
    if( difference > 0) {
        int numberOfTextAnalyserInAMonitorTime = monitorWindowDimension / 2;
        double scaleUpNumber_ = static_cast<double>(difference) / 
                                static_cast<double>(numberOfTextAnalyserInAMonitorTime);
        int scaleUpNumber = ceil(scaleUpNumber_);
        for(int i = 1; i <= scaleUpNumber; i++) {
            string command = "/home/shejialuo/Projects/Email_Microservices/scripts/TextAnalyser.sh textanalyser"
                             + to_string(i + numberOfTextAnalyser);
            shellExecute(command);
            time_t now_time=time(NULL);  
            tm*  t_tm = localtime(&now_time);  
            string stt = asctime(t_tm);
            stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
            std::string logString = stt + " " + "创建了TextAnalyser" + to_string(i + numberOfTextAnalyser);
            io_mutex.lock();
            std::cout << logString << std::endl;
            io_mutex.unlock();
        }
        numberOfTextAnalyser += scaleUpNumber;
    }
    else {
        vector<int> textAnalyserQuery_;
        for(auto &query: textAnalyserQuery) {
            string stringSplited = stringSplit(query, 6, ' ');
            textAnalyserQuery_.push_back(stringFindNumber(stringSplited));
        }
        sort(textAnalyserQuery_.begin(),textAnalyserQuery_.end());
        bool flag = true;
        int n = textAnalyserQuery_.at(textAnalyserQuery_.size() -1) - textAnalyserQuery_.at(0);
        if(n > 100) {
            flag = false;
            int scaleNumber = n / 100;
            for(int i = 1; i <= scaleNumber; ++i) {
                string command = "/home/shejialuo/Projects/Email_Microservices/scripts/TextAnalyser.sh textanalyser"
                             + to_string(i + numberOfTextAnalyser);
                shellExecute(command);
                time_t now_time=time(NULL);  
                tm*  t_tm = localtime(&now_time);  
                string stt = asctime(t_tm);
                stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
                std::string logString = stt + " " + "创建了TextAnalyser" + to_string(i + numberOfTextAnalyser);
                io_mutex.lock();
                std::cout << logString << std::endl;
                io_mutex.unlock();
            }
            numberOfTextAnalyser += scaleNumber;
        }
        if(!flag) return;
        int messageNumber = numberOfMessagesInAMonitorWindow.at(index);
        double numberRequired_ = static_cast<double>(messageNumber) / 
                              static_cast<double>(monitorWindowDimension / 2);
        int numberRequired = ceil(numberRequired_);
        while(numberRequired < numberOfTextAnalyser) {
            Client newClient("192.168.88.110",8000,8001);
            newClient.socket();
            newClient.bind();
            newClient.connect();
            newClient.send("disconnect");
            newClient.close();
            string command = "sudo docker container rm -f textanalyser" + to_string(numberOfTextAnalyser);
            shellExecute(command);
            time_t now_time=time(NULL);  
            tm*  t_tm = localtime(&now_time);  
            string stt = asctime(t_tm);
            stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
            std::string logString = stt + " " + "删除了TextAnalyser" + to_string(numberOfTextAnalyser);
            io_mutex.lock();
            std::cout << logString << std::endl;
            io_mutex.unlock();
            numberOfTextAnalyser--;
        }
    }
}

void Monitor::scaleSentimentAnalyser(string& dataBaseQuery, vector<string>& sentimentAnalyserQuery) {
    string resultSplited = stringSplit(dataBaseQuery, 7,' ');
    int arriveNumber = stringFindNumber(resultSplited);
    int difference = messageCounter - arriveNumber;
    io_mutex.lock();
    cout << "目前未到达的SentimentAnalyserResults数目为：" << difference << endl;
    io_mutex.unlock();
    if( difference > 0) {
        int numberOfSentimentAnalyserInAMonitorTime = monitorWindowDimension / 4;
        double scaleUpNumber_ = static_cast<double>(difference) / 
                                static_cast<double>(numberOfSentimentAnalyserInAMonitorTime);
        int scaleUpNumber = ceil(scaleUpNumber_);
        for(int i = 1; i <= scaleUpNumber; i++) {
            string command = "/home/shejialuo/Projects/Email_Microservices/scripts/SentimentAnalyser.sh sentimentanalyser"
                             + to_string(i + numberOfSentimentAnalyser);
            shellExecute(command);
            time_t now_time=time(NULL);  
            tm*  t_tm = localtime(&now_time);  
            string stt = asctime(t_tm);
            stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
            std::string logString = stt + " " + "创建了SentimentAnalyser" + to_string(i + numberOfSentimentAnalyser);
            io_mutex.lock();
            std::cout << logString << std::endl;
            io_mutex.unlock();
        }
        numberOfSentimentAnalyser += scaleUpNumber;
    }
    else {
        vector<int> sentimentAnalyserQuery_;
        for(auto &query: sentimentAnalyserQuery) {
            string stringSplited = stringSplit(query, 6, ' ');
            sentimentAnalyserQuery_.push_back(stringFindNumber(stringSplited));
        }
        sort(sentimentAnalyserQuery_.begin(),sentimentAnalyserQuery_.end());
        bool flag = true;
        int n = sentimentAnalyserQuery_.at(sentimentAnalyserQuery_.size() - 1) - sentimentAnalyserQuery_.at(0);
        if(n > 50) {
            flag = false;
            int scaleNumber = n / 50;
            for(int i = 1; i<=scaleNumber; ++i) {
                string command = "/home/shejialuo/Projects/Email_Microservices/scripts/SentimentAnalyser.sh sentimentanalyser"
                                + to_string(i + numberOfSentimentAnalyser);
                shellExecute(command);
                time_t now_time=time(NULL);  
                tm*  t_tm = localtime(&now_time);  
                string stt = asctime(t_tm);
                stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
                std::string logString = stt + " " + "创建了SentimentAnalyser" + to_string(i + numberOfSentimentAnalyser);
                io_mutex.lock();
                std::cout << logString << std::endl;
                io_mutex.unlock();
            }
            numberOfSentimentAnalyser += scaleNumber;
        }
        if(!flag) return;
        int messageNumber = numberOfMessagesInAMonitorWindow.at(index);
        double numberRequired_ = static_cast<double>(messageNumber) / 
                              static_cast<double>(monitorWindowDimension / 4);
        int numberRequired = ceil(numberRequired_);
        while(numberRequired < numberOfSentimentAnalyser) {
            Client newClient("192.168.90.110",8000,8001);
            newClient.socket();
            newClient.bind();
            newClient.connect();
            newClient.send("disconnect");
            newClient.close();
            string command = "sudo docker container rm -f sentimentanalyser" + to_string(numberOfSentimentAnalyser);
            shellExecute(command);
            time_t now_time=time(NULL);  
            tm*  t_tm = localtime(&now_time);  
            string stt = asctime(t_tm);
            stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
            std::string logString = stt + " " + "删除了SentimentAnalyser" + to_string(numberOfSentimentAnalyser);
            io_mutex.lock();
            std::cout << logString << std::endl;
            io_mutex.unlock();
            numberOfSentimentAnalyser--;
        }
    }
}

void Monitor::scaleLinkAnalyser(string& dataBaseQuery, vector<string>& linkAnalyserQuery) {
    string resultSplited = stringSplit(dataBaseQuery, 7,' ');
    string result = stringSplit(resultSplited, 0, '_');
    int arriveNumber = stringFindNumber(result);
    int difference = messageCounter - arriveNumber;
    io_mutex.lock();
    cout << "目前未到达的LinkAnalyserResults数目为：" << difference << endl;
    io_mutex.unlock();
    if( difference > 0) {
        int numberOfLinkAnalyserInAMonitorTime = monitorWindowDimension / 5;
        double scaleUpNumber_ = static_cast<double>(difference) / 
                                static_cast<double>(numberOfLinkAnalyserInAMonitorTime);
        int scaleUpNumber = ceil(scaleUpNumber_);
        for(int i = 1; i <= scaleUpNumber; i++) {
            string command = "/home/shejialuo/Projects/Email_Microservices/scripts/LinkAnalyser.sh linkanalyser"
                             + to_string(i + numberOfLinkAnalyser);
            shellExecute(command);
            time_t now_time=time(NULL);  
            tm*  t_tm = localtime(&now_time);  
            string stt = asctime(t_tm);
            stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
            std::string logString = stt + " " + "创建了LinkAnalyser" + to_string(i + numberOfLinkAnalyser);
            io_mutex.lock();
            std::cout << logString << std::endl;
            io_mutex.unlock();
        }
        numberOfLinkAnalyser += scaleUpNumber;
    }
    else {
        vector<int> linkAnalyserQuery_;
        for(auto &query: linkAnalyserQuery) {
            string stringSplited = stringSplit(query, 6, ' ');
            string result = stringSplit(stringSplited, 0, '_');
            linkAnalyserQuery_.push_back(stringFindNumber(result));
        }
        sort(linkAnalyserQuery_.begin(),linkAnalyserQuery_.end());
        bool flag = true;
        int n = linkAnalyserQuery_.at(linkAnalyserQuery_.at(linkAnalyserQuery_.size() - 1)) - linkAnalyserQuery_.at(0);
        if (n > 40) {
            flag = false;
            int scaleNumber = n / 40;
            for(int i = 1 ; i <= scaleNumber; ++i) {
                string command = "/home/shejialuo/Projects/Email_Microservices/scripts/LinkAnalyser.sh linkanalyser"
                                + to_string(i + numberOfLinkAnalyser);
                shellExecute(command);
                time_t now_time=time(NULL);  
                tm*  t_tm = localtime(&now_time);  
                string stt = asctime(t_tm);
                stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
                std::string logString = stt + " " + "创建了LinkAnalyser" + to_string(i + numberOfLinkAnalyser);
                io_mutex.lock();
                std::cout << logString << std::endl;
                io_mutex.unlock();
            }
            numberOfLinkAnalyser += scaleNumber;
        }
        if(!flag) return;
        int messageNumber = numberOfMessagesInAMonitorWindow.at(index);
        double numberRequired_ = static_cast<double>(messageNumber) / 
                                static_cast<double>(monitorWindowDimension / 5);
        int numberRequired = ceil(numberRequired_);
        while(numberRequired < numberOfLinkAnalyser) {
            Client newClient("192.168.86.110",8000,8001);
            newClient.socket();
            newClient.bind();
            newClient.connect();
            newClient.send("disconnect");
            newClient.close();
            string command = "sudo docker container rm -f linkanalyser" + to_string(numberOfLinkAnalyser);
            shellExecute(command);
            time_t now_time=time(NULL);  
            tm*  t_tm = localtime(&now_time);  
            string stt = asctime(t_tm);
            stt.erase(remove(stt.begin(), stt.end(), '\n'), stt.end());
            std::string logString = stt + " " + "删除了LinkAnalyser" + to_string(numberOfLinkAnalyser);
            io_mutex.lock();
            std::cout << logString << std::endl;
            io_mutex.unlock();
            numberOfLinkAnalyser--;
        }
    }
}

void Monitor::run() {
    while(true) {
        sleep(monitorWindowDimension);
        messageCounter += numberOfMessagesInAMonitorWindow.at(index);

        string command = "sudo docker logs database | grep \"Headers\" | tail -1";
        string headerAnalyserDataBaseQuery = dataBaseQuery(command);

        command = "sudo docker logs saloadbalancer | tail -1";
        string textAnalyserSaQuery = dataBaseQuery(command);

        command = "sudo docker logs database | grep \"Texts\" | tail -1";
        string sentimentAnalyserDataBaseQuery = dataBaseQuery(command);

        command = "sudo docker logs database | grep \"Links\" | tail -1";
        string linkAnalyserDataBaseQuery = dataBaseQuery(command);

        auto headeranalyserQuery = headerAnalyserQuery();
        auto textanalyserQuery = textAnalyserQuery();
        auto sentimentanalyserQuery = sentimentAnalyserQuery();
        auto linkanalyserQuery = linkAnalyserQuery();

        auto ScaleHeaderAnalyser = async(launch::async,
                                        [&]{scaleHeaderAnalyser(headerAnalyserDataBaseQuery, headeranalyserQuery);});
        auto ScaleTextAnalyser = async(launch::async, 
                                        [&]{scaleTextAnalyser(textAnalyserSaQuery, textanalyserQuery);});
        auto ScaleSentimentAnalyser = async(launch::async, 
                                        [&]{scaleSentimentAnalyser(sentimentAnalyserDataBaseQuery, sentimentanalyserQuery);});
        auto ScaleLinkAnalyser = async(launch::async, 
                                        [&]{scaleLinkAnalyser(linkAnalyserDataBaseQuery, linkanalyserQuery);});
        index = (index + 1) % numberOfMessagesInAMonitorWindow.size();
    }
}
