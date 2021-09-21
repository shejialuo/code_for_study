#ifndef LINKSANALYSIS_HPP
#define LINKSANALYSIS_HPP

#include<string>

struct LinksAnalysis {
    std::string laResults;
    std::string laMessageId;
    LinksAnalysis(std::string results, std::string messageId) {
        laResults = results;
        laMessageId = messageId;
    }
    LinksAnalysis() {}
};

#endif // LINKSANALYSIS_HPP