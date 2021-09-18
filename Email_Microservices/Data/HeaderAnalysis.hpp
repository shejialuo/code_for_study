#ifndef HEADERANALYSIS_HPP
#define HEADERANALYSIS_HPP

#include<string>

struct HeaderAnalysis {
    std::string haResults;
    std::string haMessagedId;
    HeaderAnalysis(std::string results, std::string messageId) {
        haResults = results;
        haMessagedId = messageId;
    }
    HeaderAnalysis() {}
};

#endif // HEADERANALYSIS_HPP
